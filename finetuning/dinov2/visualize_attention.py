# Copyright (c) Facebook, Inc. and its affiliates.
# Modify by Huy H. Nguyen
# Licensed under the Apache License, Version 2.0 (the 'License');
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an 'AS IS' BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import argparse
import cv2
import random
import colorsys
import requests
from io import BytesIO

from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from pathlib import Path

from model import ForensicsTransformer

# must disable xformers
os.environ['XFORMERS_DISABLED'] = '1'

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('--model', default='dinov2_vitl14_reg', type=str, help='model name')
    parser.add_argument('--outf', default='checkpoints/depth_11/dinov2_l14_reg', help='folder to output images and model checkpoints')
    parser.add_argument('--id', type=int, default=59, help='checkpoint ID')

    # parser.add_argument('--image_path', default='input/real/864_839_5.jpg', type=str, help='Path of the image to load.')
    # parser.add_argument('--image_path', default='input/real/873_872_4.jpg', type=str, help='Path of the image to load.')
    # parser.add_argument('--image_path', default='input/real/978_975_4.jpg', type=str, help='Path of the image to load.')
    # parser.add_argument('--image_path', default='input/real/987_938_5.jpg', type=str, help='Path of the image to load.')

    # parser.add_argument('--image_path', default='input/fake/864_839_5.jpg', type=str, help='Path of the image to load.')
    # parser.add_argument('--image_path', default='input/fake/873_872_4.jpg', type=str, help='Path of the image to load.')
    # parser.add_argument('--image_path', default='input/fake/978_975_4.jpg', type=str, help='Path of the image to load.')
    # parser.add_argument('--image_path', default='input/fake/987_938_5.jpg', type=str, help='Path of the image to load.')

    parser.add_argument('--image_path', default='input/fake/P2.png', type=str, help='Path of the image to load.')
    # parser.add_argument('--image_path', default='input/fake/LDM.png', type=str, help='Path of the image to load.')

    parser.add_argument('--image_size', default=(224, 224), type=int, nargs='+', help='Resize image.')
    parser.add_argument('--output_dir', default='output', help='Path where to save visualizations.')

    args = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # forensics model
    #---------------------------------------
    model = ForensicsTransformer(model_name=args.model, use_torch_hub=False)
    model.load_state_dict(torch.load(os.path.join(args.outf,'model_' + str(args.id) + '.pt'), map_location=device))
    
    patch_size = model.dino.patch_size
    num_register_tokens = model.dino.num_register_tokens
    #---------------------------------------


    # vision model
    #---------------------------------------
    # import dinov2.hub.backbones as backbones
    # import sys
    # sys.path.append('dinov2')
    # model = backbones.dinov2_vitl14_reg()

    # patch_size = model.patch_size
    # num_register_tokens = model.num_register_tokens
    #---------------------------------------

    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    # open image
    if args.image_path is None:
        # user has not specified any image - we use our own image
        print('Please use the `--image_path` argument to indicate the path of the image you wish to visualize.')
        print('Since no image path have been provided, we take the first image in our paper.')
        response = requests.get('https://dl.fbaipublicfiles.com/dino/img.png')
        img = Image.open(BytesIO(response.content))
        img = img.convert('RGB')
    elif os.path.isfile(args.image_path):
        with open(args.image_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
    else:
        print(f'Provided image path {args.image_path} is non valid.')
        sys.exit(1)
    
    #---------------------------------------
    # crop image to cut the mask
    min_size = min(img.size)
    img = img.crop((0, 0, min_size, min_size))
    #---------------------------------------
        
    transform_fwd = pth_transforms.Compose([
        pth_transforms.Resize(args.image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    img = transform_fwd(img)

    unnormalize = pth_transforms.Normalize((-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225), (1 / 0.229, 1 / 0.224, 1 / 0.225))

    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    if hasattr(model, 'dino'):
        attentions = model.dino.get_last_selfattention(img.to(device))
    else:
        attentions = model.get_last_selfattention(img.to(device))

    # remove register tokens
    attentions = attentions[:, :, num_register_tokens:, num_register_tokens:]

    nh = attentions.shape[1] # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode='nearest')[0]

    # save attentions heatmaps
    os.makedirs(args.output_dir, exist_ok=True)
    img = unnormalize(img).squeeze(dim=0).permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    attention_avg = torch.mean(attentions, axis=0).cpu().numpy()
    attention_norm = (attention_avg-np.min(attention_avg))/(np.max(attention_avg)-np.min(attention_avg))
    attention_norm = (attention_norm*255).astype(np.uint8)
    heatmap_img = cv2.applyColorMap(attention_norm, cv2.COLORMAP_JET)
    
    blended = cv2.addWeighted(heatmap_img, 0.5, img, 0.5, 0, dtype=cv2.CV_8U)

    cv2.imwrite(os.path.join(args.output_dir, 'attention.png'), attention_norm)
    cv2.imwrite(os.path.join(args.output_dir, 'blended.png'), blended)

    # torchvision.utils.save_image(torchvision.utils.make_grid(attention_avg, normalize=True, scale_each=True), os.path.join(args.output_dir, 'attention.png'))

    # for j in range(nh):
    #     fname = os.path.join(args.output_dir, 'attn-head' + str(j) + '.png')
    #     plt.imsave(fname=fname, arr=attentions[j]cpu().numpy(), format='png')
    #     print(f'{fname} saved.')
