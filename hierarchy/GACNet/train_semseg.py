import argparse
import os
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from collections import defaultdict
from torch.autograd import Variable
from S3DISDataLoader import S3DISDataLoader, recognize_all_data,class2label
import torch.nn.functional as F
import datetime
import logging
from pathlib import Path
from utils import test_seg
from tqdm import tqdm
from model import GACNet
import warnings
warnings.filterwarnings("ignore")

seg_classes = class2label
seg_label_to_cat = {}
for i,cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def parse_args():
    parser = argparse.ArgumentParser('GACNet')
    parser.add_argument('--batchSize', type=int, default=24, help='input batch size [default: 24]')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers [default: 4]')
    parser.add_argument('--epoch', type=int, default=200, help='number of epochs for training [default: 200]')
    parser.add_argument('--log_dir', type=str, default='logs/',help='decay rate of learning rate')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training [default: 0.001 for Adam, 0.01 for SGD]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay for Adam')
    parser.add_argument('--optimizer', type=str, default='SGD', help='type of optimizer')
    parser.add_argument('--multi_gpu', type=str, default=None, help='whether use multi gpu training')
    parser.add_argument('--dropout', type=float, default=0, help='dropout [default: 0]')
    parser.add_argument('--alpha', type=float, default=0.2, help='alpha for leakyRelu [default: 0.2]')
    return parser.parse_args()

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu if args.multi_gpu is None else '0,1,2,3'
    '''CREATE DIR'''
    experiment_dir = Path('./experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) +'/'+ str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath(args.log_dir)
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger('GACNet')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + '/train_GACNet.txt')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('PARAMETER ...')
    logger.info(args)
    print('Load data...')
    train_data, train_label, test_data, test_label = recognize_all_data(test_area = 5)

    dataset = S3DISDataLoader(train_data,train_label)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                             shuffle=True, num_workers=int(args.workers))
    test_dataset = S3DISDataLoader(test_data,test_label)
    testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8,
                                                 shuffle=True, num_workers=int(args.workers))

    num_classes = 13
    blue = lambda x: '\033[94m' + x + '\033[0m'
    model = GACNet(num_classes,args.dropout,args.alpha)

    if args.pretrain is not None:
        model.load_state_dict(torch.load(args.pretrain))
        print('load model %s'%args.pretrain)
        logger.info('load model %s'%args.pretrain)
    else:
        print('Training from scratch')
        logger.info('Training from scratch')
    pretrain = args.pretrain
    init_epoch = int(pretrain[-14:-11]) if args.pretrain is not None else 0

    def adjust_learning_rate(optimizer, step):
        """Sets the learning rate to the initial LR decayed by 30 every 20000 steps"""
        lr = args.learning_rate * (0.3 ** (step // 20000))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )

    '''GPU selection and multi-GPU'''
    if args.multi_gpu is not None:
        device_ids = [int(x) for x in args.multi_gpu.split(',')]
        torch.backends.cudnn.benchmark = True
        model.cuda(device_ids[0])
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    else:
        model.cuda()

    history = defaultdict(lambda: list())
    best_acc = 0
    best_meaniou = 0
    step = 0

    for epoch in range(init_epoch,args.epoch):
        for i, data in tqdm(enumerate(dataloader, 0),total=len(dataloader),smoothing=0.9):
            points, target = data
            points, target = Variable(points.float()), Variable(target.long())
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            model = model.train()
            pred = model(points[:,:3,:],points[:,3:,:])
            pred = pred.contiguous().view(-1, num_classes)
            target = target.view(-1, 1)[:, 0]
            loss = F.nll_loss(pred, target)
            history['loss'].append(loss.cpu().data.numpy())
            loss.backward()
            optimizer.step()
            step += 1
            adjust_learning_rate(optimizer, step)

        # if epoch % 10 == 0:
        #     train_metrics, train_hist_acc, cat_mean_iou = test_seg(model, dataloader,seg_label_to_cat)
        #     print('Epoch %d  %s loss: %f accuracy: %f  meanIOU: %f' % (
        #         epoch, blue('train'), history['loss'][-1], train_metrics['accuracy'],np.mean(cat_mean_iou)))
        #     logger.info('Epoch %d  %s loss: %f accuracy: %f  meanIOU: %f' % (
        #         epoch, 'train', history['loss'][-1], train_metrics['accuracy'],np.mean(cat_mean_iou)))
        #

        test_metrics, test_hist_acc, cat_mean_iou = test_seg(model, testdataloader, seg_label_to_cat)
        mean_iou = np.mean(cat_mean_iou)

        print('Epoch %d  %s accuracy: %f  meanIOU: %f' % (
                 epoch, blue('test'), test_metrics['accuracy'],mean_iou))
        logger.info('Epoch %d  %s accuracy: %f  meanIOU: %f' % (
                 epoch, 'test', test_metrics['accuracy'],mean_iou))
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            torch.save(model.state_dict(), '%s/GACNet_%.3d_%.4f.pth' % (checkpoints_dir, epoch, best_acc))
            logger.info(cat_mean_iou)
            logger.info('Save model..')
            print('Save model..')
            print(cat_mean_iou)
        if mean_iou > best_meaniou:
            best_meaniou = mean_iou
        print('Best accuracy is: %.5f'%best_acc)
        logger.info('Best accuracy is: %.5f'%best_acc)
        print('Best meanIOU is: %.5f'%best_meaniou)
        logger.info('Best meanIOU is: %.5f'%best_meaniou)


if __name__ == '__main__':
    args = parse_args()
    main(args)
