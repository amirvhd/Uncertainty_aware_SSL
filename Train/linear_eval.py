import sys
import time
import torch
from utils.util import AverageMeter, accuracy
from utils.util import warmup_learning_rate
from models.resnet_big import SupConResNet, LinearClassifier
import torch.backends.cudnn as cudnn
from torch import nn
from sklearn.metrics import classification_report, balanced_accuracy_score


class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def set_model_linear(model_name, number_cls, path, nh=5):
    model = SupConResNet(name=model_name, n_heads=nh)
    criterion = torch.nn.CrossEntropyLoss()
    classifier = LinearClassifier(name=model_name, num_classes=number_cls)

    ckpt = torch.load(path)
    state_dict = ckpt

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = MyDataParallel(model)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)

    return model, classifier, criterion


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    schedular = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        if opt.pu:
            images = images.reshape(images.shape[0], 3, 32, 32).float()
        # warm-up learning rate
        # warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

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
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()
    schedular.step()
    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    idx, len(val_loader), batch_time=batch_time,
                    loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def evaluate(val_loader, model, classifier, opt):
    model.eval()
    classifier.eval()
    with torch.no_grad():
        true_y, pred_y = [], []
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            output = classifier(model.encoder(images))
            true_y.extend(labels.cpu())
            pred_y.extend(torch.argmax(output, dim=1).cpu())
        #print(classification_report(true_y, pred_y, digits=3))
        #print(balanced_accuracy_score(true_y, pred_y))
        return classification_report(true_y, pred_y)


def predict(dataloader, model, laplace=False):
    py = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for x, _ in dataloader:
            if laplace:
                py.append(model(x.to(device)))
            else:
                # py.append(torch.softmax(model(x.to(device)), dim=-1))
                py.append(model(x.to(device)))
        res = torch.cat(py).cpu().detach().numpy()

    return res
