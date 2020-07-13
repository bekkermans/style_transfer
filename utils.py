import torch.nn as nn
import torch.nn.functional as F
import torch
import cv2
import numpy as np
import math
import numbers


class RGBtoYUV(nn.Module):

    def __init__(self):
        super(RGBtoYUV, self).__init__()
        self.convert_tensor = torch.tensor([[0.29900, -0.16874, 0.50000],
                                            [0.58700, -0.33126, -0.41869],
                                            [0.11400, 0.50000, -0.08131]])

    def forward(self, image):
        image = image.squeeze(0)
        image = torch.matmul(self.convert_tensor.to(image.device) / 255., image)
        image[1:, ...] += 128.0 / 255.
        image = image.unsqueeze(0)
        return image


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, padding=2, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = padding
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
          torch.exp((-((mgrid - mean) / std) ** 2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)


class LaplacianPyramid(nn.Module):

    def __init__(self, depth, channel=3, kernel_size=5, sigma=1):
        super(LaplacianPyramid, self).__init__()
        self.depth = depth
        self.filter = GaussianSmoothing(channel, kernel_size, sigma)

    def forward(self, image):

        # Gausian pyramid
        g_pyramid = [image]
        for i in range(self.depth):
            image = self.filter(image)
            image = F.interpolate(image, (image.shape[2] // 2, image.shape[3] // 2), mode='bilinear',
                                  align_corners=False)
            g_pyramid.append(image)

        # Laplacian pyrammid
        l_pyramid = [g_pyramid[-1]]
        for i in range(self.depth, 0, -1):
            b, c, h, w = g_pyramid[i - 1].shape
            gaus = F.interpolate(g_pyramid[i], (h, w), mode='bilinear', align_corners=False)
            gaus = self.filter(gaus)
            lap = torch.sub(g_pyramid[i - 1], gaus)
            l_pyramid.append(lap)
        return l_pyramid

    def reconstruct(self, l_pyramid):
        image = l_pyramid[0]
        for i in range(1, len(l_pyramid)):
            b, c, h, w = l_pyramid[i].shape
            image = F.interpolate(image, (h, w), mode='bilinear', align_corners=False)
            image = self.filter(image)
            image = torch.add(l_pyramid[i], image)
        return image


class CreateSpatialTensor(nn.Module):

    def __init__(self):
        super(CreateSpatialTensor, self).__init__()

    def forward(self, features, indices):
        h, w = features[0].shape[2], features[0].shape[3]
        result_list = []
        for f in features:
            f = F.interpolate(f, (h, w), mode='bilinear', align_corners=False)
            result_list.append(torch.index_select(f.view(f.shape[0], f.shape[1], -1), 2, indices))
        result_tensor = torch.cat([feat.contiguous() for feat in result_list], 1)
        return result_tensor


class GetRandomIndices(nn.Module):

    def __init__(self, device):
        super(GetRandomIndices, self).__init__()
        self.generator = np.random.default_rng()
        self.device = device

    def forward(self, shape, num=1024):
        b, c, h, w = shape
        indices = self.generator.choice(h * w, size=num, replace=False, shuffle=True)
        return torch.as_tensor(indices, dtype=torch.long, device=self.device)


def numpy_to_tensor(image):
    image = np.transpose(image, axes=(2, 0, 1))
    image = torch.as_tensor(image, dtype=torch.float32)
    return image.unsqueeze(0)


def load_image(path, size=1024):
    image = cv2.imread(filename=path)
    if image is None:
        raise RuntimeError(f'Error: Couldn\'t load file {path}')

    height, width, channel = image.shape

    if size < 256:
        raise RuntimeError(f'Error: Resolution too small. Image can\'t be smaller than 256 px')
    if channel < 3 or channel > 3:
        raise RuntimeError(f'Error: Grayscale image currently doesnt support')

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = (cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC) / 255.)
    return numpy_to_tensor(image)


def save_tensor_to_image(im_tensor, filename, dsize=512):
    image = im_tensor.detach().cpu().numpy()
    im_max = np.max(image)
    im_min = np.min(image)
    image = (image - im_min) / (im_max - im_min)
    new_style = np.transpose(np.squeeze(image, axis=0), axes=(1, 2, 0)) * 255
    new_style = cv2.resize(new_style, dsize=(dsize, dsize))
    cv2.imwrite(filename, cv2.cvtColor(new_style, cv2.COLOR_RGB2BGR))
