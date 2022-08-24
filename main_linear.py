from __future__ import print_function

import argparse
import time
import math
import torch
import copy

from Dataloader.dataloader import set_loader, data_loader
from utils.util import adjust_learning_rate
from utils.util import set_optimizer

# from laplace import Laplace
from Train.linear_eval import set_model_linear, train, evaluate, predict, validate
# import tensorboard_logger as tb_logger
from torch.utils.tensorboard import SummaryWriter

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

"""
tensorboard --logdir=~/DATA/loggings_cifar10_models_UAloss/linear_evaluation/exp1 --port 6006
on your pc:
ssh -N -f -L localhost:16006:localhost:6006 ra49bid2@datalab2.srv.lrz.de
on browser:
http://localhost:16006
"""


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=10,
                        help='number of heads')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'isic'], help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='./saved_models/cifar10_models/simclr_cifar10_0_epoch200.pt',
                        help='path to pre-trained model')
    parser.add_argument('--semi', action='store_true',
                        help='semi-supervised')
    parser.add_argument('--pu', action='store_true',
                        help='positive unlabeled mode')
    parser.add_argument('--semi_percent', type=int, default=10,
                        help='percentage of data usage in semi-supervised')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='number of ensemble models')
    parser.add_argument('--nh', type=int, default=1,
                        help='number of heads')
    parser.add_argument('--lamda', type=int, default=1,
                        help='number of heads')
    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = '../../DATA2/'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.tb_path = '../../DATA/loggings_{}_models_UAloss/linear_evaluation/exp1'.format(opt.dataset)

    # warm-up for large-batch training,
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'svhn':
        opt.n_cls = 10
    elif opt.dataset == 'isic':
        opt.n_cls = 7
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    opt.model_path = './saved_models/{}_linear_models'.format(opt.dataset)
    opt.classifier_path = './saved_models/{}_linear_models'.format(opt.dataset)
    return opt


def main():
    best_acc = 0
    opt = parse_option()
    writer = SummaryWriter(log_dir=opt.tb_path)
    best_epoch = 0
    # build data loader
    train_loader, val_loader, test_loader, _ = data_loader(dataset=opt.dataset, batch_size=opt.batch_size)
    # train_loader, val_loader = set_loader(dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers)
    ensemble = opt.ensemble
    for i in range(ensemble):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        opt.ckpt = (
            './saved_models/{}_models_UAloss/simclr_{}_{}_epoch800_{}heads_{}.pt'.format(opt.dataset, opt.dataset, i,
                                                                                      opt.nh, opt.lamda))
        # build model and criterion
        model, classifier, criterion = set_model_linear(model_name=opt.model, number_cls=opt.n_cls, path=opt.ckpt)
        # tensorboard
        # logger = tb_logger.Logger(logdir=opt.tb_path, flush_secs=2)
        # build optimizer
        optimizer = set_optimizer(opt, classifier)
        print('ensemble number is {}:'.format(i))
        # training routine
        best_classifier = None
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, classifier, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('ensemble number {}, Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(i,
                                                                                                  epoch, time2 - time1,
                                                                                                  acc))

            # eval for one epoch
            val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('accuracy', val_acc, epoch)
            writer.add_scalar("Loss/eval", val_loss, epoch)

            if val_acc > best_acc:
                #   best_epoch = epoch
                best_acc = val_acc
                best_classifier = copy.deepcopy(classifier)
            # if epoch - best_epoch > opt.patience:
            #  break
        evaluate(test_loader, model, best_classifier, opt)
        print('best accuracy: {:.2f}'.format(best_acc))
        writer.flush()
        writer.close()
        # save the last model
        if opt.semi:

            torch.save(best_classifier.state_dict(),
                       './saved_models/{}_models/simclr_linear_{}_epoch{}_percent{}.pt'.format(opt.dataset, i,
                                                                                               opt.epochs,
                                                                                               opt.semi_percent))
            torch.save(model.state_dict(),
                       './saved_models/{}_models/simclr_encoder_{}_epoch{}_percent{}.pt'.format(opt.dataset, i,
                                                                                                opt.epochs,
                                                                                                opt.semi_percent))
        else:
            torch.save(best_classifier.state_dict(),
                       './saved_models/{}_models_UAloss/simclr800_linear_{}_epoch{}_{}heads_{}.pt'.format(opt.dataset,
                                                                                                          i,
                                                                                                          opt.epochs,
                                                                                                          opt.nh,
                                                                                                          opt.lamda))
            torch.save(model.state_dict(),
                       './saved_models/{}_models_UAloss/simclr800_encoder_{}_epoch{}_{}heads_{}.pt'.format(opt.dataset,
                                                                                                           i,
                                                                                                           opt.epochs,
                                                                                                           opt.nh,
                                                                                                           opt.lamda))


if __name__ == '__main__':
    main()
