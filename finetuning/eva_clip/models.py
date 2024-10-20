"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
"""

import torch
import torch.nn as nn
from eva_clip.factory import create_model_and_transforms

class EvaClip(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, num_blocks, model_name="EVA02-CLIP-L-14", pretrained="eva_clip"):
        super(EvaClip, self).__init__()
        self.num_blocks = num_blocks
        self.net, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        self.total_blocks = len(self.net.visual.blocks)
        self.stop_block = self.total_blocks - num_blocks

        self.net.visualnorm = nn.Identity()
        self.net.visual.head = nn.Identity()
        self.net.text = nn.Identity()

    def forward(self, x):
        x = self.net.visual.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.net.visual.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.net.visual.pos_embed is not None:
            x = x + self.net.visual.pos_embed
        x = self.net.visual.pos_drop(x)
        t = self.net.visual.patch_dropout(x)

        rel_pos_bias = self.net.visual.rel_pos_bias() if self.net.visual.rel_pos_bias is not None else None

        for i in range(0, self.stop_block):
            t = self.net.visual.blocks[i](t, rel_pos_bias=rel_pos_bias)

        res = []
        for i in range(self.stop_block, self.total_blocks):
            t = self.net.visual.blocks[i](t, rel_pos_bias=rel_pos_bias)
            res += [t]
        
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

        for i in range( self.num_blocks):
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

    def forward(self, x):
        temp = x[0].reshape(x[0].shape[0], -1)
        y = self.block_linears.linear_0(temp)

        for i in range(1, self.num_blocks):
            temp = x[i].reshape(x[i].shape[0], -1)
            z = getattr(self.block_linears, f'linear_{i}')(temp)
            y = torch.cat((y, z), dim=-1)

        return self.final_linear(y)


class ForensicsTransformer(nn.Module):
    def __init__(self, model_name="EVA02-CLIP-L-14", pretrained="eva_clip", num_unfrozen_blocks=1, hidden_dim=1024, num_labels=2):
        super(ForensicsTransformer, self).__init__()

        self.num_unfrozen_blocks = num_unfrozen_blocks
        
        net, _, preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        self.clip = net.visual
        del net, preprocess

        self.clip.Linear = nn.Linear(hidden_dim, num_labels)

        # freeze gradient of clip.patch_embed
        for param in self.clip.patch_embed.parameters():
            param.requires_grad = False

        self.clip.pos_embed.requires_grad = False

        num_blocks = len(self.clip.blocks)
        
        # freeze gradients of first n - k clip.blocks
        for i in range(num_blocks - self.num_unfrozen_blocks):
            for param in self.clip.blocks[i].parameters():
                param.requires_grad = False

        # unfrozen parameters:
        # self.clip.cls_token

        # init self.clip.linear layer
        self.clip.Linear.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.clip(x)