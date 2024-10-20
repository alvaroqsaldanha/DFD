"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
"""


import sys
sys.path.append('mae')
import torch
import torch.nn as nn
from mae import models_vit

class MAE(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, num_blocks, model_name='vit_large_patch16', pretrained="mae/pretrained/mae_pretrain_vit_large.pth"):
        super(MAE, self).__init__()
        self.num_blocks = num_blocks
        self.net = getattr(models_vit, model_name)()
        self.net.head = nn.Identity()

        self.net.load_state_dict(torch.load(pretrained)['model'])
        self.total_blocks = len(self.net.blocks)
        self.stop_block = self.total_blocks - num_blocks

    def forward(self, x):
        B = x.shape[0]
        x = self.net.patch_embed(x)

        cls_tokens = self.net.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.net.pos_embed
        x = self.net.pos_drop(x)

        for i in range(0, self.stop_block):
            x = self.net.blocks[i](x)

        res = []
        for i in range(self.stop_block, self.total_blocks):
            x = self.net.blocks[i](x)
            res += [x]
        
        return res

class LinearClassifierConcatAll(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_blocks, hidden_dim=1024, dropout=0.0, num_labels=2):
        super(LinearClassifierConcatAll, self).__init__()
        self.num_labels = num_labels
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.block_linears = nn.ParameterList()
        self.final_linear = nn.Linear(hidden_dim*num_blocks, num_labels)

        for i in range(self.num_blocks):
            linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(dim, hidden_dim),
                nn.ReLU()
            )
            setattr(self.block_linears, f'linear_{i}', linear)

        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        temp = x[0].reshape(x[0].shape[0], -1)
        y = self.block_linears.linear_0(temp)

        for i in range(1, self.num_blocks):
            temp = x[i].reshape(x[i].shape[0], -1)
            z = getattr(self.block_linears, f'linear_{i}')(temp)
            y = torch.cat((y, z), dim=-1)

        return self.final_linear(y)


class ForensicsTransformer(nn.Module):
    def __init__(self, model_name='vit_large_patch16', pretrained="mae/pretrained/mae_pretrain_vit_large.pth", num_unfrozen_blocks=1, hidden_dim=1024, num_labels=2, img_size=224, pretrained_21k=True):
        super(ForensicsTransformer, self).__init__()

        self.num_unfrozen_blocks = num_unfrozen_blocks
        
        self.mae = getattr(models_vit, model_name)()
        self.mae.head = nn.Identity()
        self.mae.load_state_dict(torch.load(pretrained)['model'])
        
        self.mae.linear = nn.Linear(hidden_dim, num_labels)

        # freeze gradient of mae.patch_embed
        for param in self.mae.patch_embed.parameters():
            param.requires_grad = False

        self.mae.pos_embed.requires_grad = False

        num_blocks = len(self.mae.blocks)
        
        # freeze gradients of first n - k mae.blocks
        for i in range(num_blocks - self.num_unfrozen_blocks):
            for param in self.mae.blocks[i].parameters():
                param.requires_grad = False

        # unfrozen parameters:
        # self.mae.cls_token
        
        # init self.mae.linear layer
        self.mae.head.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.mae(x)