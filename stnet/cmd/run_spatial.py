import torch
import torchvision
import numpy as np
import logging
import pathlib
import traceback
import random
import time
import os
import stnet
import glob
import socket

def run_spatial(args=None):

    stnet.utils.logging.setup_logging(args.logfile, args.loglevel)
    logger = logging.getLogger(__name__)
    try:
        ### Log information about run ###
        logger.info(args)
        logger.info("Configuration file: {}".format(stnet.config.FILENAME))
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            logger.info("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        else:
            logger.info("CUDA_VISIBLE_DEVICES not defined.")
        logger.info("CPUs: {}".format(os.sched_getaffinity(0)))
        logger.info("GPUs: {}".format(torch.cuda.device_count()))
        logger.info("Hostname: {}".format(socket.gethostname()))

        ### Seed RNGs ###
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        ### Select device for computation ###
        device = ("cuda" if args.gpu else "cpu")

        ### Split patients into folds ###
        patient = stnet.utils.util.get_spatial_patients()

        train_patients = []
        test_patients = []
        n_test = round(args.test * len(patient))
        is_test = [True for i in range(n_test)] + [False for i in range(len(patient) - n_test)]
        random.shuffle(is_test)

        for (i, p) in enumerate(patient):
            if args.trainpatients is None and args.testpatients is None:
                if is_test[i]:
                    for s in patient[p]:
                        test_patients.append((p, s))
                else:
                    for s in patient[p]:
                        train_patients.append((p, s))
            elif args.trainpatients is None and args.testpatients is not None:
                for s in patient[p]:
                    if p in args.testpatients or (p, s) in args.testpatients:
                        test_patients.append((p, s))
                    else:
                        train_patients.append((p, s))
            elif args.trainpatients is not None and args.testpatients is None:
                for s in patient[p]:
                    if p in args.trainpatients or (p, s) in args.trainpatients:
                        train_patients.append((p, s))
                    else:
                        test_patients.append((p, s))
            else:
                for s in patient[p]:
                    if p in args.trainpatients or (p, s) in args.trainpatients:
                        train_patients.append((p, s))
                    if p in args.testpatients or (p, s) in args.testpatients:
                        test_patients.append((p, s))

        ### Dataset setup ###
        window = args.window
        if args.window_raw is not None:
            window = args.window_raw
        train_dataset = stnet.datasets.Spatial(train_patients, window=window, gene_filter=args.gene_filter, downsample=args.downsample, norm=args.norm, gene_transform=args.gene_transform, transform=torchvision.transforms.ToTensor(), feature=(args.model == "rf"))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=args.gpu)

        # Estimate mean and covariance
        t = time.time()
        n_samples = 10
        mean = 0.
        std = 0.
        n = 0
        for (i, (X, *_)) in enumerate(train_loader):
            X = X.transpose(0, 1).contiguous().view(3, -1)
            n += X.shape[1]
            mean += torch.sum(X, dim=1)
            std += torch.sum(X ** 2, dim=1)
            if i > n_samples:
                break
        mean /= n
        std = torch.sqrt(std / n - mean ** 2)
        logger.info("Estimating mean (" + str(mean) + ") and std (" + str(std) + " took " + str(time.time() - t))

        transform = []
        if args.window_raw is not None:  # TODO: and not equal to window
            transform.append(torchvision.transforms.RandomCrop((args.window, args.window)))
        if args.brightness != 0 or args.contrast != 0 or args.saturation != 0 or args.hue:
            transform.append(torchvision.transforms.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue))
        transform.extend([torchvision.transforms.RandomHorizontalFlip(),
                          torchvision.transforms.RandomVerticalFlip(),
                          torchvision.transforms.RandomApply([torchvision.transforms.RandomRotation((90, 90))]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize(mean=mean, std=std)])
        transform = torchvision.transforms.Compose(transform)

        train_dataset.transform = transform

        if args.average:
            transform = torchvision.transforms.Compose([stnet.transforms.EightSymmetry(),
                                                        torchvision.transforms.Lambda(lambda symmetries: torch.stack([torchvision.transforms.Normalize(mean=mean, std=std)(torchvision.transforms.ToTensor()(s)) for s in symmetries]))])
        else:
            transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=mean, std=std)])
        # TODO: random crops on test too?
        test_dataset = stnet.datasets.Spatial(test_patients, transform, window=args.window, gene_filter=args.gene_filter, downsample=args.downsample, norm=args.norm, gene_transform=args.gene_transform, feature=(args.model == "rf"))

        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch, num_workers=args.workers, shuffle=True, pin_memory=args.gpu)

        # Find number of required outputs
        if args.task == "tumor":
            outputs = 2
        elif args.task == "gene":
            outputs = train_dataset[0][2].shape[0]
        elif args.task == "geneb":
            outputs = 2 * train_dataset[0][2].shape[0]
        elif args.task == "count":
            outputs = 1

        ### Model setup ###
        if args.model == "rf":
            pass
        elif args.model == "inception_v3":
            model = torchvision.models.__dict__[args.model](pretrained=args.pretrained, aux_logits=False)
        else:
            model = torchvision.models.__dict__[args.model](pretrained=args.pretrained)

        start_epoch = 0
        if args.model != "linear" and args.model != "rf":
            # Replace last layer
            # TODO: if loading weights, should just match outs
            stnet.utils.nn.set_out_features(model, outputs)

            if args.gpu:
                model = torch.nn.DataParallel(model)
            model.to(device)


            ### Optimizer setup ###
            parameters = stnet.utils.nn.get_finetune_parameters(model, args.finetune, args.randomize)
            optim = torch.optim.__dict__[args.optim](parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


            if args.load is not None:
                model.load_state_dict(torch.load(args.load)["model"])

            ### Reload parameters from incomplete run ###
            if args.restart:
                for epoch in range(args.epochs)[::-1]:
                    if os.path.isfile(args.checkpoint + str(epoch + 1) + ".pt"):
                        start_epoch = epoch + 1
                        try:
                            checkpoint = torch.load(args.checkpoint + str(epoch + 1) + ".pt")
                        except:
                            # A runtime error is thrown if the checkpoint is corrupted (most likely killed while saving)
                            # Continue checking for earlier checkpoints
                            logger.info("Detected corrupted checkpoint at epoch #{}. Skipping checkpoint.".format(start_epoch))
                            continue
                        model.load_state_dict(checkpoint["model"])
                        optim.load_state_dict(checkpoint["optim"])
                        logger.info("Detected run stopped at epoch #{}. Restarting from checkpoint.".format(start_epoch))
                        break

        # Compute mean expression for initial params in model and as baseline
        if args.task == "gene" or args.task == "geneb" or args.task == "count":
            t = time.time()

            mean_expression = torch.zeros(train_dataset[0][2].shape)
            mean_expression_tumor = torch.zeros(train_dataset[0][2].shape)
            mean_expression_normal = torch.zeros(train_dataset[0][2].shape)
            tumor = 0
            normal = 0
            load_image = train_loader.dataset.load_image
            train_loader.dataset.load_image = False  # Temporarily stop loading images to save time
            for (i, (_, y, gene, *_)) in enumerate(train_loader):
                print("{:8d} / {:d}:    {:4.0f} / {:4.0f} seconds".format(i + 1, len(train_loader), time.time() - t, (time.time() - t) * len(train_loader) / (i + 1)), end="\r", flush=True)
                mean_expression += torch.sum(gene, 0)
                mean_expression_tumor += torch.sum(y.float() * gene, 0)
                mean_expression_normal += torch.sum((1 - y).float() * gene, 0)
                tumor += torch.sum(y).detach().numpy()
                normal += torch.sum(1 - y).detach().numpy()
            train_loader.dataset.load_image = load_image

            mean_expression /= float(tumor + normal)
            mean_expression_tumor /= float(tumor)
            mean_expression_normal /= float(normal)
            median_expression = torch.log(1 + torch.Tensor(train_dataset.median_expression))

            mean_expression = mean_expression.to(device)
            mean_expression_tumor = mean_expression_tumor.to(device)
            mean_expression_normal = mean_expression_normal.to(device)
            median_expression = median_expression.to(device)

            if args.model != "rf":
                m = model
                if args.gpu:
                    m = m.module

                if (isinstance(m, torchvision.models.AlexNet) or
                    isinstance(m, torchvision.models.VGG)):
                    last = m.classifier[-1]
                elif isinstance(m, torchvision.models.DenseNet):
                    last = m.classifier
                elif (isinstance(m, torchvision.models.ResNet) or
                      isinstance(m, torchvision.models.Inception3)):
                    last = m.fc
                else:
                    raise NotImplementedError()

                if args.load is None and start_epoch == 0:
                    last.weight.data.zero_()
                    if args.task == "gene":
                        last.bias.data = mean_expression.clone()
                    elif args.task == "geneb":
                        last.bias.data.zero_()
                    elif args.task == "count":
                        mean_expression = torch.sum(mean_expression, 0, keepdim=True)
                        mean_expression_tumor = torch.sum(mean_expression_tumor, 0, keepdim=True)
                        mean_expression_normal = torch.sum(mean_expression_normal, 0, keepdim=True)
                        last.bias.data = mean_expression.clone()
                    else:
                        raise ValueError()

                if args.gene_mask is not None:
                    args.gene_mask = torch.Tensor([args.gene_mask])
                    args.gene_mask = args.gene_mask.to(device)

                logger.info("Computing mean expression took {}".format(time.time() - t))

        ### Training Loop ###
        for epoch in range(start_epoch, args.epochs):
            logger.info("Epoch #" + str(epoch + 1))
            for (dataset, loader) in [("train", train_loader), ("test", test_loader)]:

                t = time.time()
                torch.set_grad_enabled(dataset == "train")
                if args.model != "rf":
                    model.train(dataset == "train")

                total = 0
                total_mean = 0
                total_type = 0
                correct = 0
                positive = 0
                mse = np.zeros(train_dataset[0][2].shape)
                mse_mean = np.zeros(train_dataset[0][2].shape)
                mse_type = np.zeros(train_dataset[0][2].shape)
                features = []
                genes = []
                predictions = []
                counts = []
                tumor = []
                coord = []
                patient = []
                section = []
                pixel = []

                save_pred = (dataset == "test" and args.pred_root is not None and (args.save_pred_every is None or (epoch + 1) % args.save_pred_every == 0))

                n = 0
                logger.info(dataset + ":")
                for (i, (X, y, gene, c, ind, pat, s, pix, f)) in enumerate(loader):
                    if save_pred:
                        counts.append(gene.detach().numpy())
                        tumor.append(y.detach().numpy())
                        coord.append(c.detach().numpy())
                        patient += pat
                        section += s
                        pixel.append(pix.detach().numpy())

                    X = X.to(device)
                    y = y.to(device)
                    gene = gene.to(device)

                    if dataset == "test" and args.average:
                        batch, n_sym, c, h, w = X.shape
                        X = X.view(-1, c, h, w)
                    if args.model == "rf":
                        if dataset == "train":
                            features.append(f.detach().numpy())
                            genes.append(gene.cpu().detach().numpy())
                            pred = gene
                        else:
                            pred = torch.Tensor(model.predict(f.detach().numpy())).to(device)
                    else:
                        pred = model(X)
                    if dataset == "test" and args.average:
                        pred = pred.view(batch, n_sym, -1).mean(1)
                    if save_pred:
                        predictions.append(pred.cpu().detach().numpy())

                    if args.task == "tumor":
                        y = torch.squeeze(y, dim=1)
                        loss = torch.nn.functional.cross_entropy(pred, y, reduction='sum')
                        correct += torch.sum(torch.argmax(pred, dim=1) == y).cpu().detach().numpy()
                        positive += torch.sum(y).cpu().detach().numpy()
                    elif args.task == "gene":
                        if args.gene_mask is None:
                            loss = torch.sum((pred - gene) ** 2) / outputs
                        else:
                            loss = torch.sum(args.gene_mask * (pred - gene) ** 2) / torch.sum(args.gene_mask)
                        mse += torch.sum((pred - gene) ** 2, 0).cpu().detach().numpy()

                        # Evaluating baseline performance
                        total_mean += (torch.sum((mean_expression - gene) ** 2) / outputs).cpu().detach().numpy()
                        mse_mean   += torch.sum((mean_expression - gene) ** 2, 0).cpu().detach().numpy()
                        y = y.float()
                        total_type += (torch.sum((y * mean_expression_tumor + (1 - y) * mean_expression_normal - gene) ** 2) / outputs).cpu().detach().numpy()
                        mse_type += torch.sum((y * mean_expression_tumor + (1 - y) * mean_expression_normal - gene) ** 2, 0).cpu().detach().numpy()
                    elif args.task == "geneb":
                        gene = (gene > torch.log(1 + median_expression)).type(torch.int64)
                        # TODO: gene mask needs to work here too
                        loss = torch.nn.functional.cross_entropy(pred.reshape(-1, 2), gene.reshape(-1), reduction='sum') / outputs
                    elif args.task == "count":
                        gene = torch.sum(gene, 1, keepdim=True)
                        loss = torch.sum((pred - gene) ** 2)
                        total_mean += torch.sum((mean_expression - gene) ** 2).cpu().detach().numpy()
                        total_type += torch.sum((y * mean_expression_tumor + (1 - y) * mean_expression_normal - gene) ** 2).cpu().detach().numpy()
                    else:
                        raise ValueError()
                    total += loss.cpu().detach().numpy()
                    n += y.shape[0]

                    message = ""
                    message += "{:8d} / {:d} ({:4.0f} / {:4.0f}):".format(i + 1, len(loader), time.time() - t, (time.time() - t) * len(loader) / (i + 1))
                    message += "    Loss={:.3f}".format(total / n)
                    if args.task == "tumor":
                        message += "    Accuracy={:.3f}    Tumor={:.3f}".format(correct / n, positive / n)
                    if args.task == "gene":
                        message += "    MSE={:.3f}    Type:{:.3f}".format(total_mean / n, total_type / n)
                    logger.debug(message)

                    if dataset == "train" and (args.model != "rf"):
                        optim.zero_grad()
                        loss.backward()
                        optim.step()
                logger.info("    Loss:       " + str(total / len(loader.dataset)))
                if args.task == "tumor":
                    logger.info("    Accuracy:   " + str(correct / len(loader.dataset)))
                    logger.info("    Percentage: " + str(positive / len(loader.dataset)))
                if args.task == "gene":
                    logger.info("    MSE:        " + str(total_mean / len(loader.dataset)))
                    logger.info("    Type:       " + str(total_type / len(loader.dataset)))
                    logger.info("    Best:       " + str(max((mse_mean - mse) / mse_mean)))
                    logger.info("    Worst:      " + str(min((mse_mean - mse) / mse_mean)))
                # TODO: debug messages for geneb and count are incomplete (also in the progress bar)

                if dataset == "train" and args.model == "rf":
                    features = np.concatenate(features)
                    genes = np.concatenate(genes)
                    if args.model == "linear":
                        import sklearn
                        model = sklearn.linear_model.LinearRegression().fit(features, genes)
                    else:
                        import sklearn.ensemble
                        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100).fit(features, genes)

                if save_pred:
                    predictions = np.concatenate(predictions)
                    counts = np.concatenate(counts)
                    tumor = np.concatenate(tumor)
                    coord = np.concatenate(coord)
                    pixel = np.concatenate(pixel)

                    if args.task == "tumor":
                        me = None
                        me_tumor = None
                        me_normal = None
                    else:
                        me = mean_expression.cpu().numpy(),
                        me_tumor = mean_expression_tumor.cpu().numpy(),
                        me_normal = mean_expression_normal.cpu().numpy(),

                    pathlib.Path(os.path.dirname(args.pred_root)).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(args.pred_root + str(epoch + 1),
                                        task=args.task,
                                        tumor=tumor,
                                        counts=counts,
                                        predictions=predictions,
                                        coord=coord,
                                        patient=patient,
                                        section=section,
                                        pixel=pixel,
                                        mean_expression=me,
                                        mean_expression_tumor=me_tumor,
                                        mean_expression_normal=me_normal,
                                        ensg_names=test_dataset.ensg_names,
                                        gene_names=test_dataset.gene_names,
                    )

                # Saving after test so that if information from test is needed, they will not get skipped
                if dataset == "test" and args.checkpoint is not None and ((epoch + 1) % args.checkpoint_every) == 0 and args.model != "rf":
                    pathlib.Path(os.path.dirname(args.checkpoint)).mkdir(parents=True, exist_ok=True)
                    # TODO: if model is on gpu, does loading automatically put on gpu?
                    # https://discuss.pytorch.org/t/how-to-store-model-which-trained-by-multi-gpus-dataparallel/6526
                    # torch.save(model.state_dict(), args.checkpoint + str(epoch + 1) + ".pt")
                    torch.save({
                        'model': model.state_dict(),
                        'optim' : optim.state_dict(),
                    }, args.checkpoint + str(epoch + 1) + ".pt")

                    if epoch != 0 and (args.keep_checkpoints is None or (epoch + 1 - args.checkpoint_every) not in args.keep_checkpoints):
                        os.remove(args.checkpoint + str(epoch + 1 - args.checkpoint_every) + ".pt")

    except Exception as e:
        logger.exception(traceback.format_exc())
        raise
