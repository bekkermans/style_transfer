import torch
import torch.nn.functional as F
import argparse
import sys
import math

from torch.optim import RMSprop
from utils import load_image, LaplacianPyramid, GetRandomIndices, CreateSpatialTensor, save_tensor_to_image
from model import VggEncoder
from loss import StyleLoss


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("content", type=str, help='content image for stylization')
    parser.add_argument("style", type=str, help='style image')
    parser.add_argument("--alpha", type=float, default=16.0, help='strength of stylization')
    parser.add_argument("--output", type=str, default="result.jpg", help='output filename')
    parser.add_argument("--max_resolution", type=int, default=512, help='output image resolution')
    args = parser.parse_args()

    vgg_encoder = VggEncoder().to(device)
    pyramid = LaplacianPyramid(5).to(device)
    select_tensor = CreateSpatialTensor().to(device)
    indices_generator = GetRandomIndices(device)

    try:
        content_image = load_image(args.content, args.max_resolution)
        style_image = load_image(args.style, args.max_resolution)
    except Exception as e:
        print(f'Error: {e}')
        sys.exit(1)

    # Calculating depth of laplacian pyramid
    pyr_depth = int(math.log2(content_image.shape[-1])) - 6 + 1
    image_pyramid = LaplacianPyramid(pyr_depth).to(device)
    origin_image_pyramid = image_pyramid(content_image.to(device))

    criteria = StyleLoss()
    alpha = args.alpha
    content_pyramid = None

    for j, con_image in enumerate(origin_image_pyramid[1:]):
        batch, channel, height, width = con_image.shape
        print(f'Image size: ({height}, {width})')

        org_con_image = F.interpolate(content_image, (height, width), mode='bilinear', align_corners=False).to(device)
        st_image = F.interpolate(style_image, (height, width), mode='bilinear', align_corners=False).to(device)

        with torch.no_grad():
            indices = indices_generator(con_image.shape)
            content_features = vgg_encoder(org_con_image)
            vgg_style_features = vgg_encoder(st_image)
            style_features = select_tensor(vgg_style_features, indices)
            for i in range(5):
                indices = indices_generator(con_image.shape)
                style_features = torch.cat((style_features, select_tensor(vgg_style_features, indices)), 2)

        if j == 0:
            step_image = torch.add(con_image, st_image.mean(dim=(2, 3), keepdim=True))
            lr = 2e-3

        elif j < pyr_depth - 1:
            step_image = torch.add(con_image,
                                   F.interpolate(pyramid.reconstruct(content_pyramid), (height, width),
                                                 mode='bilinear', align_corners=False))
            lr = 2e-3
        else:
            step_image = F.interpolate(pyramid.reconstruct(content_pyramid), (height, width),
                                       mode='bilinear', align_corners=False)
            lr = 1e-3

        content_pyramid = pyramid(step_image)
        content_pyramid = [layer.data.requires_grad_() for layer in content_pyramid]
        optim = RMSprop(content_pyramid, lr=lr)
        try:
            for i in range(200):
                result_image = pyramid.reconstruct(content_pyramid)
                optim.zero_grad()
                out_features = vgg_encoder(result_image)
                out_features = select_tensor(out_features, indices)
                con_features = select_tensor(content_features, indices)
                loss = criteria(out_features, con_features, style_features, alpha)
                loss.backward()
                optim.step()
                indices = indices_generator(con_image.shape)
        except RuntimeError as e:
            print(f'Error: {e}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
        del out_features, con_features, content_features, vgg_style_features
        alpha /= 2.0
    result = pyramid.reconstruct(content_pyramid)
    result.data.clamp_(0, 1)
    save_tensor_to_image(result, args.output, args.max_resolution)
    print('Done!')
