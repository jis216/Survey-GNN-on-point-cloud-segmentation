#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""

"""


from torch_geometric.data import DataLoader
from torch_geometric.datasets import (
    ShapeNet,
    ModelNet,
    S3DIS
)


def load_dataset(
    path, transform=None, pre_transform=None, pre_filter=None,
    category=None, name='10', test_area=6
):
    if path.name == 'ShapeNet':
        train_dataset = ShapeNet(path, category, split='trainval',
                                 transform=transform, pre_transform=pre_transform,
                                 pre_filter=pre_filter)
        test_dataset  = ShapeNet(path, category, split='test',
                                 transform=transform, pre_transform=pre_transform,
                                 pre_filter=pre_filter)
    elif path.name == 'ModelNet':
        train_dataset = ModelNet(path, name=name, train=True,
                                 transform=transform, pre_transform=pre_transform,
                                 pre_filter=pre_filter)
        test_dataset  = ModelNet(path, name=name, train=False,
                                 transform=transform, pre_transform=pre_transform,
                                 pre_filter=pre_filter)
    elif path.name == 'S3DIS':
        train_dataset = S3DIS(path, test_area=test_area, train=True,
                              transform=transform, pre_transform=pre_transform,
                              pre_filter=pre_filter)
        test_dataset  = S3DIS(path, test_area=test_area, train=False,
                              transform=transform, pre_transform=pre_transform,
                              pre_filter=pre_filter)

    return train_dataset, test_dataset


def load_dataloader(train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True,
                              num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False,
                             num_workers=6)

    return train_loader, test_loader
