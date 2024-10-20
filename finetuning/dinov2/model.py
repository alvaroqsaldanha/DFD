"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
"""

import torch
import torch.nn as nn

class ForensicsTransformer(nn.Module):
    def __init__(self, model_name, num_unfrozen_blocks=1, hidden_dim=1024, num_labels=2, use_torch_hub=True):
        super(ForensicsTransformer, self).__init__()

        self.num_unfrozen_blocks = num_unfrozen_blocks
        if use_torch_hub:
            self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            import sys
            import importlib
            sys.path.append('dinov2')
            vit_constructor = importlib.import_module('dinov2.hub.backbones')
            self.dino = getattr(vit_constructor, model_name)(pretrained=True)
        
        self.dino.linear = nn.Linear(hidden_dim, num_labels)

        # freeze gradient of dino.patch_embed
        for param in self.dino.patch_embed.parameters():
            param.requires_grad = False

        self.dino.pos_embed.requires_grad = False
        self.dino.mask_token.requires_grad = False

        num_blocks = len(self.dino.blocks)

        # freeze gradients of first n - k dino.blocks
        for i in range(num_blocks - self.num_unfrozen_blocks):
            for param in self.dino.blocks[i].parameters():
                param.requires_grad = False

        # unfrozen parameters:
        # self.dino.cls_token
        # self.dino.register_tokens
        
        # init self.dino.linear layer
        self.dino.linear.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.dino(x)