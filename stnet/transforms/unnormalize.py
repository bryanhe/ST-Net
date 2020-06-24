import torch
import torchvision

def unnormalize(tensor, mean, std, inplace=False):
    """Unnormalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    Returns:
        Tensor: Normalized Tensor image.
    """
    # TODO: figure out where this function is
    # if not _is_tensor_image(tensor):
    #     raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    mean = torch.as_tensor(mean, dtype=torch.float32, device=tensor.device)
    std = torch.as_tensor(std, dtype=torch.float32, device=tensor.device)
    tensor.mul_(std[:, None, None]).add_(mean[:, None, None])
    return tensor

class Unnormalize(object):
    """Unnormalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return unnormalize(tensor, self.mean, self.std, self.inplace)


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
