"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
"""

import torch
import torch.nn as nn
import timm

class EfficientNetv2(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, model_name="tf_efficientnetv2_l.in21k_ft_in1k", pretrained=True, num_unfrozen_blocks=1, hidden_dim=1280, num_labels=2):
        super(EfficientNetv2, self).__init__()
        self.num_unfrozen_blocks = num_unfrozen_blocks

        self.net = timm.create_model(model_name, pretrained=pretrained)
        self.net.classifier = nn.Linear(hidden_dim, num_labels)

        for param in self.net.conv_stem.parameters():
            param.requires_grad = False

        for param in self.net.bn1.parameters():
            param.requires_grad = False

        for i in range(0, len(self.net.blocks) - num_unfrozen_blocks):
            for param in self.net.blocks[i].parameters():
                param.requires_grad = False
        
        self.net.classifier.apply(self.init_weights)

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x):
        return self.net(x)

class EfficientNetv2Backbone(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, num_blocks, model_name="tf_efficientnetv2_l.in21k_ft_in1k", pretrained=True):
        super(EfficientNetv2Backbone, self).__init__()

        self.net = timm.create_model(model_name, pretrained=pretrained)
        self.num_blocks = num_blocks
        self.total_blocks = len(self.net.blocks)
        self.normal_blocks = self.total_blocks - self.num_blocks

        for param in self.net.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        ret = []

        x = self.net.conv_stem(x)
        x = self.net.bn1(x)

        for i in range(0, self.normal_blocks):
            x = self.net.blocks[i](x)

        for i in range(self.normal_blocks, self.total_blocks):
            x = self.net.blocks[i](x)
            ret.append(x.reshape(x.shape[0], -1))

        return ret


class LinearClassifierConcat(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, num_blocks, dims=[110592, 129024, 55296, 92160], hidden_dim=1024, dropout=0.0, num_labels=2):
        super(LinearClassifierConcat, self).__init__()
        self.num_labels = num_labels
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim
        self.block_linears = nn.ParameterList()
        self.final_linear = nn.Linear(hidden_dim*num_blocks, num_labels)

        for i in range( self.num_blocks):
            linear = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(dims[i], hidden_dim),
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
        y = self.block_linears.linear_0(x[0])

        for i in range(1, self.num_blocks):
            z = getattr(self.block_linears, f'linear_{i}')(x[i])
            y = torch.cat((y, z), dim=-1)

        return self.final_linear(y)