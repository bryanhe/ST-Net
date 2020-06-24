import stnet
import torch
import torchvision
import numpy as np
import scipy

def features(x):
    import skimage
    import histomicstk as htk
    if len(x.shape) == 3:
        x = x.reshape(1, *x.shape)
    n, c, h, w = x.shape

    mean = np.array([0.54, 0.51, 0.68])
    std = np.array([0.25, 0.21, 0.16])
    im_input = stnet.transforms.Unnormalize(mean=mean, std=std)(x).to("cpu")
    im_input = np.swapaxes(255 * im_input.numpy(), 1, 3)

    # Based on https://digitalslidearchive.github.io/HistomicsTK/examples/nuclei-segmentation.html#Load-input-image
    # Strain matrix from https://digitalslidearchive.github.io/HistomicsTK/histomicstk.preprocessing.color_deconvolution.html
    W = np.array([[0.650, 0.072, 0], [0.704, 0.990, 0], [0.286, 0.105, 0]])
    # W = (W - mean) / std
    # print(W)
    
    count = torch.empty(n, 1)
    area = torch.empty(n, 1)
    for i in range(n):
        # torchvision.transforms.ToPILImage()(stnet.transforms.Unnormalize(mean=mean, std=std)(x).to("cpu")[i, :, :, :]).save("temp/{}_raw.jpg".format(i))
        # perform standard color deconvolution
        # im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(torchvision.transforms.ToPILImage()(x[i, :, :, :]), W).Stains
        im_stains = htk.preprocessing.color_deconvolution.color_deconvolution(im_input[i, :, :, :], W).Stains
        
        # # Display results
        # plt.figure(figsize=(20, 10))
        # 
        # plt.subplot(1, 2, 1)
        # plt.imshow(im_stains[:, :, 0])
        # plt.title("Hematoxylin")
        # 
        # plt.subplot(1, 2, 2)
        # plt.imshow(im_stains[:, :, 1])
        # _ = plt.title("Eosin")
        # plt.savefig("temp/{}_sep.jpg".format(i))

        # get nuclei/hematoxylin channel
        im_nuclei_stain = im_stains[:, :, 0]
        
        # segment foreground
        foreground_threshold = 60
        
        im_fgnd_mask = scipy.ndimage.morphology.binary_fill_holes(
            im_nuclei_stain < foreground_threshold)

        # run adaptive multi-scale LoG filter
        min_radius = 10
        max_radius = 15
        
        im_log_max, im_sigma_max = htk.filters.shape.cdog(
            im_nuclei_stain, im_fgnd_mask,
            sigma_min=min_radius * np.sqrt(2),
            sigma_max=max_radius * np.sqrt(2)
        )
        
        # detect and segment nuclei using local maximum clustering
        local_max_search_radius = 10
        
        im_nuclei_seg_mask, seeds, maxima = htk.segmentation.nuclear.max_clustering(
            im_log_max, im_fgnd_mask, local_max_search_radius)
        
        # filter out small objects
        min_nucleus_area = 80
        
        im_nuclei_seg_mask = htk.segmentation.label.area_open(
            im_nuclei_seg_mask, min_nucleus_area).astype(np.int)

        # compute nuclei properties
        objProps = skimage.measure.regionprops(im_nuclei_seg_mask)

        count[i] = len(objProps)
        area[i] = sum(map(lambda x: x.area, objProps)) / h / w
        
        # print('Number of nuclei = ', len(objProps))
        # print(objProps[0])

        # # Display results
        # plt.figure(figsize=(20, 10))
        # 
        # plt.subplot(1, 2, 1)
        # print(im_nuclei_seg_mask.shape)
        # print(im_input[i, :, :, :].shape)
        # plt.imshow(skimage.color.label2rgb(im_nuclei_seg_mask, im_input[i, :, :, :], bg_label=0), origin='lower')
        # plt.title('Nuclei segmentation mask overlay')
        # 
        # plt.subplot(1, 2, 2)
        # plt.imshow(im_input[i, :, :, :])
        # plt.xlim([0, im_input.shape[1]])
        # plt.ylim([0, im_input.shape[0]])
        # plt.title('Nuclei bounding boxes')
        # 
        # for i in range(len(objProps)):
        # 
        #     c = [objProps[i].centroid[1], objProps[i].centroid[0], 0]
        #     width = objProps[i].bbox[3] - objProps[i].bbox[1] + 1
        #     height = objProps[i].bbox[2] - objProps[i].bbox[0] + 1
        # 
        #     cur_bbox = {
        #         "type":        "rectangle",
        #         "center":      c,
        #         "width":       width,
        #         "height":      height,
        #     }
        # 
        #     plt.plot(c[0], c[1], 'g+')
        #     mrect = matplotlib.patches.Rectangle([c[0] - 0.5 * width, c[1] - 0.5 * height] ,
        #                                width, height, fill=False, ec='g', linewidth=2)
        #     plt.gca().add_patch(mrect)
        # plt.savefig("temp/{}_box.jpg".format(i))


    mean = x.reshape(n, c, -1).mean(2)
    std = x.reshape(n, c, -1).std(2)
    return torch.cat((mean, std, count, area), 1)
