import torch
import torch.nn as nn

from utils import RGBtoYUV


class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()
        self.rgb_to_yuv = RGBtoYUV()

    def cosine_distance(self, x, y):
        d = x.shape[1]
        x = x.contiguous().view(d, -1).T
        y = y.contiguous().view(d, -1)
        x_norm = torch.sqrt(torch.pow(x, 2).sum(1).view(-1, 1))
        y_norm = torch.sqrt(torch.pow(y, 2).sum(0).view(1, -1))
        dist = 1. - torch.mm(x, y) / x_norm / y_norm
        return dist

    def euclidean_distance(self, x, y):
        d = x.shape[1]
        x = x.contiguous().view(d, -1).T
        y = y.contiguous().view(d, -1)
        x_norm = torch.pow(x, 2).sum(1).view(-1, 1)
        y_norm = torch.pow(y, 2).sum(0).view(1, -1)
        dist = x_norm + y_norm - 2.0 * torch.mm(x, y)
        return torch.clamp(dist, 1e-5, 1e5) / x.shape[1]

    def moment_loss(self, extracted_features, style_features):
        x_mean = torch.mean(extracted_features, dim=2, keepdim=False)
        y_mean = torch.mean(style_features, dim=2, keepdim=False)
        x = extracted_features.squeeze(0) - x_mean.T
        y = style_features.squeeze(0) - y_mean.T
        x_cov = torch.mm(x, x.T) / (x.shape[1] - 1)
        y_cov = torch.mm(y, y.T) / (y.shape[1] - 1)
        return torch.abs(x_mean - y_mean).mean() + torch.abs(x_cov - y_cov).mean()

    def remd(self, x, y):
        dim = x.shape[1]
        if dim > 3:
            cost_matrix = self.cosine_distance(x, y)
        else:
            cost_matrix = torch.add(self.cosine_distance(x, y), torch.sqrt(self.euclidean_distance(x, y)))
        min1, _ = torch.min(cost_matrix, 0)
        min2, _ = torch.min(cost_matrix, 1)
        return torch.max(min1.mean(), min2.mean())

    def content_loss(self, extracted_features, vgg_content_features):
        x = self.cosine_distance(extracted_features, extracted_features)
        y = self.cosine_distance(vgg_content_features, vgg_content_features)
        return torch.abs(x - y).mean()

    def forward(self, extracted_features, content_feature, vgg_style_features, alpha):

        # Content loss
        lc = self.content_loss(extracted_features, content_feature)

        # Momentum loss
        lm = self.moment_loss(extracted_features, vgg_style_features)

        # Remd style
        lr = self.remd(extracted_features, vgg_style_features)

        # Color loss
        im_color = self.rgb_to_yuv(extracted_features[:, :3, ...])
        style_color = self.rgb_to_yuv(vgg_style_features[:, :3, ...])
        lp = self.remd(im_color, style_color) / alpha

        total_loss = (lr + lm + lp + lc * alpha) / (2. + alpha + (1.0 / alpha))
        return total_loss

