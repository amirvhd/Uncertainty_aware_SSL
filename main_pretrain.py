from __future__ import print_function

import os
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
from Train.pretrain import train, set_model
from Dataloader.dataloader import set_loader_simclr

from utils.util import adjust_learning_rate
from utils.util import set_optimizer

"""
python main.py --batch_size 512\
  --learning_rate 0.5 \
  --temp 0.5 \
  --cosine
"""

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.5,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'isic'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')

    # temperature
    parser.add_argument('--temp', type=float, default=0.5,
                        help='temperature for loss function')
    parser.add_argument('--nh', type=int, default=1,
                        help='number of heads')
    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
               and opt.mean is not None \
               and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../../DATA2/'
    opt.model_path = './saved_models/{}_models_UAloss'.format(opt.dataset)
    opt.tb_path = '../../DATA2/loggings_{}_models_UAloss_{}'.format(opt.dataset, opt.nh)
    opt.save_folder = opt.model_path
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    return opt


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader_simclr(dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                     size=opt.size)

    # print(torch.cuda.is_available())
    # build model and criterion
    ensemble = 1
    for i in range(ensemble):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        model, criterion = set_model(model_name=opt.model, temperature=opt.temp, syncBN=opt.syncBN)

        # build optimizer
        optimizer = set_optimizer(opt, model)

        # tensorboard
        logger = tb_logger.Logger(logdir=opt.tb_path, flush_secs=2)

        time1 = time.time()

        # training routine
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)
            # train for one epoch
            time3 = time.time()
            loss = train(train_loader, model, criterion, optimizer, epoch, opt)
            time4 = time.time()
            print('ensemble {}, epoch {}, total time {:.2f}'.format(i, epoch, time4 - time3))

            # tensorboard logger
            logger.log_value('loss', loss, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
            # logger.log_value('std', std_loss, epoch)
        time2 = time.time()
        print('ensemble {}, total time {:.2f}'.format(i, time2 - time1))

        save_file = os.path.join(
            opt.model_path, 'simclr_{}_{}_epoch{}_{}heads.pt'.format(opt.dataset, i, opt.epochs, opt.nh))
        torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()
