#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""

"""

import numpy as np

import torch.nn.functional as F

from tqdm import tqdm


def train(train_loader, model, optimizer, scheduler, epoch, device, writer):
    model.train()

    pbar = tqdm(enumerate(train_loader), desc='Train', total=len(train_loader), unit='batch')
    train_loss = np.array([])
    correct_nodes = total_nodes = 0
    for i, data in pbar:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        train_loss = np.append(train_loss, loss.item())
        correct_nodes += out.argmax(dim=1).eq(data.y).sum().item()
        total_nodes += data.num_nodes

        pbar.set_postfix_str(f"train_loss={np.mean(train_loss):.4f}; train_acc={correct_nodes/total_nodes:.4f}")
        writer.add_scalar(f'Epoch {epoch} Loss/Train', np.mean(train_loss), i)
