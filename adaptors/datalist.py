"""
Copyright (c) 2024, National Institute of Informatics
All rights reserved.
Author: Huy H. Nguyen
"""

import os
from PIL import Image

class DataListDataset():
    def __init__(self, data_dir, data_list, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.list = []

        #parsing data_list
        with open(data_list, 'r') as f_reader:
            data = [item.rstrip() for item in f_reader.readlines()]
           
            for item in data:
                tmp = item.split(',')
                self.list.append([tmp[0], int(tmp[1])])

    def __getitem__(self, index):

        path = os.path.join(self.data_dir, self.list[index][0])
        data = Image.open(path, mode='r').convert('RGB')

        if self.transform is not None:
            data = self.transform(data)

        return (self.list[index][0], data, int(self.list[index][1]))

    def __len__(self):
        return len(self.list)

    def name(self):
        return 'DataList Dataset'
