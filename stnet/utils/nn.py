import torch
import torchvision
import stnet

def set_out_features(model, outputs):
    """Changes number of outputs for the model.

    The change occurs in-place, but the new model is also returned."""

    if (isinstance(model, torchvision.models.AlexNet) or
        isinstance(model, torchvision.models.VGG)):
        inputs = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(inputs, outputs, bias=True)
        model.classifier[-1].weight.data.zero_()
        model.classifier[-1].bias.data.zero_()
    elif (isinstance(model, torchvision.models.ResNet) or 
          isinstance(model, torchvision.models.Inception3)):
        inputs = model.fc.in_features
        model.fc = torch.nn.Linear(inputs, outputs, bias=True)
        model.fc.weight.data.zero_()
        model.fc.bias.data.zero_()
    elif isinstance(model, torchvision.models.DenseNet):
        inputs = model.classifier.in_features
        model.classifier = torch.nn.Linear(inputs, outputs, bias=True)
        model.classifier.weight.data.zero_()
        model.classifier.bias.data.zero_()
    else:
        raise NotImplementedError()

    return model

def get_finetune_parameters(model, layers, randomize):
    if layers is None:
        return model.parameters()
    else:
        if isinstance(model, torch.nn.DataParallel):
            model = model.module 
        # if isinstance(torchvision.models.vgg.VGG):
        #     parameters = model.classifier[-4:].parameters()
        # elif isinstance(torchvision.models.densenet.DenseNet):
        #     parameters = model.classifier.parameters()
        # elif (isinstance(torchvision, torchvision.models.resnet.ResNet) or
        #       isinstance(torchvision, torchvision.models.inception.Inception3)):
        #     parameters = model.fc.parameters()
        if isinstance(model, torchvision.models.densenet.DenseNet):
            modules = [model.classifier]
            modules.extend(model.features[-(layers - 1):])
        else:
            raise NotImplementedError()

        parameters = []
        for module in modules:
            if randomize:
                for m in module.modules():
                    if isinstance(m, torch.nn.Conv2d):
                        torch.nn.init.kaiming_normal_(m.weight)
                    elif isinstance(m, torch.nn.BatchNorm2d):
                        torch.nn.init.constant_(m.weight, 1)
                        torch.nn.init.constant_(m.bias, 0)
                    elif isinstance(m, torch.nn.Linear):
                        m.reset_parameters()
                        torch.nn.init.constant_(m.bias, 0)

            parameters += list(module.parameters())
        return parameters


def set_window_size(model, window):
    """Makes minimal changes to model to handle different window size.

    The change occurs in-place, but the new model is also returned."""

    class DimensionFinder(torch.nn.Module):
        def __init__(self, out_features):
            super(DimensionFinder, self).__init__()
            self.in_features = None
            self.out_features = out_features

        def forward(self, x):
            self.in_features = x.shape[1]
            return torch.Tensor(x.shape[0], self.out_features)

        def extra_repr(self):
            return 'in_features={}, out_features={}'.format(
                self.in_features, self.out_features
            )


    if isinstance(model, torchvision.models.vgg.VGG):
        orig = model.classifier[0]
        df = DimensionFinder(orig.out_features)
        model.classifier[0] = df
    elif isinstance(model, torchvision.models.resnet.ResNet):
        orig = model.fc
        df = DimensionFinder(orig.out_features)
        model.fc = df
    elif isinstance(model, torchvision.models.densenet.DenseNet):
        orig = model.classifier
        df = DimensionFinder(orig.out_features)
        model.classifier = df
    elif isinstance(model, torchvision.models.inception.Inception3):
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    training = model.training
    model.train(False)
    model(torch.Tensor(1, 3, window, window))  # TODO: tensor should be send to the model's device
    model.train(training)

    if df.in_features == orig.in_features:
        new = orig
    else:
        new = torch.nn.Linear(df.in_features, orig.out_features)

    if isinstance(model, torchvision.models.vgg.VGG):
        model.classifier[0] = new
    elif isinstance(model, torchvision.models.resnet.ResNet):
        model.fc = new
    elif isinstance(model, torchvision.models.densenet.DenseNet):
        model.classifier = new
    elif isinstance(model, torchvision.models.inception.Inception3):
        raise NotImplementedError()
    else:
        raise NotImplementedError()

    return model

def extract_features(model):
    """Modifies the model to return (prediction, features) rather than just
    prediction.
    The change occurs in-place, but the new model is also returned."""

    if (isinstance(model, torchvision.models.AlexNet) or
        isinstance(model, torchvision.models.VGG)):
        model.classifier[-1] = InputExtractor(model.classifier[-1])
    elif (isinstance(model, torchvision.models.ResNet) or
          isinstance(model, torchvision.models.Inception3)):
        model.fc = InputExtractor(model.fc)
    elif (isinstance(model, torchvision.models.DenseNet) or
          isinstance(model, torchvision.models.MobileNetV2)):
        model.classifier = InputExtractor(model.classifier)
    else:
        raise NotImplementedError()

    return model

class Identity(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class InputExtractor(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, x):
        return self.module(x), x
