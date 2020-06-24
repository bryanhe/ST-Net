import torchvision

class EightSymmetry(object):
    """Returns a tuple of the eight symmetries resulting from rotation and reflection.
    
    This behaves similarly to TenCrop.

    This transform returns a tuple of images and there may be a mismatch in the number of inputs and targets your Dataset returns. See below for an example of how to deal with this.

    Example:
     transform = Compose([
         EightSymmetry(), # this is a tuple of PIL Images
         Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
     ])
    """

    def __call__(self, img):
        identity = lambda x: x
        ans = []
        for i in [identity, torchvision.transforms.RandomHorizontalFlip(1)]:
            for j in [identity, torchvision.transforms.RandomVerticalFlip(1)]:
                for k in [identity, torchvision.transforms.RandomRotation((90, 90))]:
                    ans.append(i(j(k(img))))
        return tuple(ans)

    def __repr__(self):
        return self.__class__.__name__ + "()"
