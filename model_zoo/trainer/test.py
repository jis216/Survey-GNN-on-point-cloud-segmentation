#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 Hengda Shi <hengda.shi@cs.ucla.edu>
#
# Distributed under terms of the MIT license.

"""

"""

import torch

from torch_geometric.utils import intersection_and_union as i_and_u

from tqdm import tqdm


@torch.no_grad()
def test(test_loader, model, device):
    model.eval()

    y_mask = test_loader.dataset.y_mask
    ious = [[] for _ in range(len(test_loader.dataset.categories))]

    pbar = tqdm(test_loader, desc='Test', total=len(test_loader), unit='batch')
    for data in pbar:
        data = data.to(device)
        pred = model(data).argmax(dim=1)

        i, u = i_and_u(pred, data.y, test_loader.dataset.num_classes, data.batch)
        iou = i.cpu().to(torch.float) / u.cpu().to(torch.float)
        iou[torch.isnan(iou)] = 1

        # Find and filter the relevant classes for each category.
        for iou, category in zip(iou.unbind(), data.category.unbind()):
            ious[category.item()].append(iou[y_mask[category]])

    # Compute mean IoU.
    ious = [torch.stack(iou).mean(0).mean(0) for iou in ious]
    return torch.tensor(ious).mean().item()
