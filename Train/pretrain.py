import sys
import time
import torch
from utils.util import AverageMeter
from utils.util import warmup_learning_rate
from models.resnet_big import SupConResNet, LinearClassifier
from utils.losses import SupConLoss
import torch.backends.cudnn as cudnn
from torch import nn


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, ((image1, image2), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # images = torch.cat([image1, image2], dim=0)
        if torch.cuda.is_available():
            image1 = image1.cuda(non_blocking=True)
            image2 = image2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute features and features std
        features, features_std = model(image1, image2)
        # compute loss
        loss = criterion(features, features_std, epoch)
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

    return losses.avg


def set_model(model_name, temperature, syncBN = False):
    model = SupConResNet(name=model_name)
    criterion = SupConLoss(temperature=temperature)

    # enable synchronized Batch Normalization

    if torch.cuda.is_available():
        model = model.cuda()
        if syncBN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion
