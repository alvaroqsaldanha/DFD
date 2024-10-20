"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LinearClassifierWS(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_blocks, num_labels=2):
        super(LinearClassifierWS, self).__init__()
        self.num_labels = num_labels
        self.num_blocks = num_blocks

        self.weights = nn.ParameterList()
        self.linear = nn.Linear(dim, num_labels)

        for i in range( self.num_blocks):
            weight = nn.Parameter(torch.FloatTensor([1.0 / self.num_blocks]))
            setattr(self.weights, f'weight_{i}', weight)

        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        data = x[0][:,0]
        y = data * self.weights.weight_0

        for i in range(1, self.num_blocks):
            data = x[i][:,0]
            y += data * getattr(self.weights, f'weight_{i}')

        return self.linear(y)
    
class LinearClassifierWSv2(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_blocks, num_labels=2):
        super(LinearClassifierWSv2, self).__init__()
        self.num_labels = num_labels
        self.num_blocks = num_blocks

        self.weights = nn.ParameterList()
        self.linear = nn.Linear(dim, num_labels)

        for i in range( self.num_blocks):
            weight = nn.Parameter(torch.FloatTensor([1.0 / self.num_blocks]))
            setattr(self.weights, f'weight_{i}', weight)

        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        data = x[0][1]
        y = data * self.weights.weight_0

        for i in range(1, self.num_blocks):
            data = x[i][1]
            y += data * getattr(self.weights, f'weight_{i}')

        return self.linear(y)

class LinearClassifierWSAll(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_blocks, num_labels=2):
        super(LinearClassifierWSAll, self).__init__()
        self.num_labels = num_labels
        self.num_blocks = num_blocks

        self.weights = nn.ParameterList()
        self.linear = nn.Linear(dim, num_labels)

        for i in range( self.num_blocks):
            weight = nn.Parameter(torch.FloatTensor([1.0 / self.num_blocks]))
            setattr(self.weights, f'weight_{i}', weight)

        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        data = x[0]
        data = data.view(data.shape[0], -1)
        y = data * self.weights.weight_0

        for i in range(1, self.num_blocks):
            data = x[i]
            data = data.view(data.shape[0], -1)
            y += data * getattr(self.weights, f'weight_{i}')

        return self.linear(y)


class LinearClassifierWSAllv2(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_blocks, num_labels=2):
        super(LinearClassifierWSAllv2, self).__init__()
        self.num_labels = num_labels
        self.num_blocks = num_blocks

        self.weights = nn.ParameterList()
        self.linear = nn.Linear(dim, num_labels)

        for i in range( self.num_blocks):
            weight = nn.Parameter(torch.FloatTensor([1.0 / self.num_blocks]))
            setattr(self.weights, f'weight_{i}', weight)

        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        concat = torch.cat((x[0][0].view(x[0][0].shape[0], -1),
                            x[0][1]
                            ), dim=-1)
        y = concat * self.weights.weight_0

        for i in range(1, self.num_blocks):
            concat = torch.cat((x[i][0].view(x[i][0].shape[0], -1),
                                x[i][1]
                                ), dim=-1)
            y += concat * getattr(self.weights, f'weight_{i}')

        return self.linear(y)

class LinearClassifierDropoutWSAllv2(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_blocks, dropout=0.0, num_labels=2):
        super(LinearClassifierDropoutWSAllv2, self).__init__()
        self.num_labels = num_labels
        self.num_blocks = num_blocks
        self.dropout = dropout

        self.weights = nn.ParameterList()
        self.linear = nn.Linear(dim, num_labels)

        for i in range( self.num_blocks):
            weight = nn.Parameter(torch.FloatTensor([1.0 / self.num_blocks]))
            setattr(self.weights, f'weight_{i}', weight)

        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, is_training=False):
        concat = torch.cat((x[0][0].view(x[0][0].shape[0], -1),
                            x[0][1]
                            ), dim=-1)
        if is_training:
            y = F.dropout(concat, p=self.dropout) * self.weights.weight_0
        else:
            y = concat * self.weights.weight_0

        for i in range(1, self.num_blocks):
            concat = torch.cat((x[i][0].view(x[i][0].shape[0], -1),
                                x[i][1]
                                ), dim=-1)
            if is_training:
                y += F.dropout(concat, p=self.dropout) * getattr(self.weights, f'weight_{i}')
            else:
                y += concat * getattr(self.weights, f'weight_{i}')

        return self.linear(y)

class LinearClassifierConcatAllv2(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_blocks, hidden_dim=1024, dropout=0.0, num_labels=2):
        super(LinearClassifierConcatAllv2, self).__init__()
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
        concat = torch.cat((x[0][0].view(x[0][0].shape[0], -1),
                            x[0][1]
                            ), dim=-1)
        y = self.block_linears.linear_0(concat)

        for i in range(1, self.num_blocks):
            concat = torch.cat((x[i][0].view(x[i][0].shape[0], -1),
                                x[i][1]
                                ), dim=-1)
            z = getattr(self.block_linears, f'linear_{i}')(concat)
            y = torch.cat((y, z), dim=-1)

        return self.final_linear(y)

class LinearClassifierSingle(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifierSingle, self).__init__()
        self.linear = nn.Linear(dim, num_labels)
        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.linear(x)
    
class LinearClassifierDropoutSingle(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, dropout=0.0, num_labels=2):
        super(LinearClassifierDropoutSingle, self).__init__()
        self.dropout = dropout
        self.linear = nn.Linear(dim, num_labels)

        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x, is_training=False):
        if is_training:
            x = F.dropout(x, p=self.dropout)
        return self.linear(x)

class MLPClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, hidden=512, num_labels=2):
        super(MLPClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_labels)
        )
        
        self.apply(self.init_weights)

    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.01)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        return self.classifier(x)
