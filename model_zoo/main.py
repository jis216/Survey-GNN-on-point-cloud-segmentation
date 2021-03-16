#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""

"""


from pathlib import Path


import torch
from torch.utils.tensorboard import SummaryWriter

import torch_geometric.transforms as T

from tqdm import trange


from models.dgcnn import DGCNN
from models.pointnet import PointNet
from trainer.train import train
from trainer.test import test
from utils.data_utils import load_dataset, load_dataloader


if __name__ == "__main__":
    category = 'Airplane'  # Pass in `None` to train on all categories.
    dataset_name = 'ShapeNet'
    #  dataset_name = 'ModelNet'
    #  model_name = 'DGCNN'
    model_name = 'PointNet'
    path = Path.cwd().absolute()/'data'/dataset_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])
    pre_transform = T.NormalizeScale()


    train_dataset, test_dataset = load_dataset(path, transform, pre_transform,
                                               category=category)
    train_loader, test_loader = load_dataloader(train_dataset, test_dataset)

    model = eval(model_name)(train_dataset.num_classes, k=30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    writer = SummaryWriter()

    pbar = trange(1, 31, desc='Epoch', unit='epoch')
    for epoch in pbar:
        train(train_loader, model, optimizer, scheduler, epoch, device, writer)
        iou = test(test_loader, model, device)
        pbar.set_postfix_str(f'test_IoU={iou}')
