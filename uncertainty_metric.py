import torch
import argparse
import torch.distributions as dists
from models.concatenate import MyEnsemble
import numpy as np
from Dataloader.dataloader import data_loader
from Train.linear_eval import set_model_linear, predict
from utils.metrics import OELoss, SCELoss, TACELoss, ACELoss, ECELoss, MCELoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'isic', 'cifar10h', 'cifar10c'], help='dataset')
    parser.add_argument('--model_path', type=str,
                        default='t', help='model path')
    parser.add_argument('--classifier_path', type=str,
                        default='', help='classifier path')
    parser.add_argument('--semi', action='store_true',
                        help='semi-supervised')
    parser.add_argument('--semi_percent', type=int, default=10,
                        help='percentage of data usage in semi-supervised')
    parser.add_argument('--nh', type=int, default=1,
                        help='number of heads')
    parser.add_argument('--lamda1', type=float, default=0,
                        help='number of heads')
    parser.add_argument('--lamda2', type=float, default=0.1,
                        help='number of heads')
    parser.add_argument('--dl', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='number of ensemble models')
    opt = parser.parse_args()
    return opt


def ensemble(n, nh, lamda1, lamda2, dataset, targets, n_cls, test_loader, dl=False, semi=False, semi_percent=10,
             model_dir=".", classifier_dir="."):
    probs_ensemble2_model = []
    if semi == True:
        for i in range(n):
            linear_model_path =model_dir
            simclr_path =classifier_dir
            model, classifier, criterion = set_model_linear("resnet50", n_cls, simclr_path, nh=nh)
            classifier.load_state_dict(torch.load(linear_model_path))
            linear_model = MyEnsemble(model.encoder, classifier).cuda().eval()
            probs_ensemble2_model.append(predict(test_loader, linear_model, laplace=False))

    else:
        for i in range(n):
            linear_model_path = model_dir
            simclr_path = classifier_dir

            model, classifier, criterion = set_model_linear("resnet50", n_cls, simclr_path, nh=nh)
            classifier.load_state_dict(torch.load(linear_model_path))
            linear_model = MyEnsemble(model.encoder, classifier).cuda().eval()
            probs_ensemble2_model.append(predict(test_loader, linear_model, laplace=False))
    probs_ensemble2_model = np.array(probs_ensemble2_model)
    probs_ensemble2 = np.mean(probs_ensemble2_model, 0)
    acc_ensemble2 = (probs_ensemble2.argmax(-1) == targets).mean()
    oe = OELoss()
    sce = SCELoss()
    ace = ACELoss()
    tace = TACELoss()
    ece = ECELoss()
    mce = MCELoss()
    oe_res = oe.loss(output=probs_ensemble2, labels=targets, logits=True)
    sce_res = sce.loss(output=probs_ensemble2, labels=targets, logits=True)
    ace_res = ace.loss(output=probs_ensemble2, labels=targets, logits=True)
    tace_res = tace.loss(output=probs_ensemble2, labels=targets, logits=True)
    ece_res = ece.loss(output=probs_ensemble2, labels=targets, logits=True, n_bins=15)
    mce_res = mce.loss(output=probs_ensemble2, labels=targets, logits=True, n_bins=5)
    nll_ensemble2 = -dists.Categorical(torch.softmax(torch.tensor(probs_ensemble2), dim=-1)).log_prob(
        torch.tensor(targets)).mean()
    results = {
        "ACCURACY": 100 * acc_ensemble2,
        "NLL": 1 * nll_ensemble2.numpy(),
        "ECE": ece_res,
        "OE": oe_res,
        "ACE": ace_res,
        "SCE": sce_res,
        "TACE": tace_res,
        "MCE": mce_res,

    }
    # print("number of ensemble is {}".format(n))
    print(
        f'[ensemble] Acc.: {acc_ensemble2:.1%}; ECE: {ece_res:.1%}; NLL: {nll_ensemble2:.3}; '
        f'OE: {oe_res:.3}; MCE: {mce_res:.3};SCE: {sce_res:.3}; ACE: {ace_res:.3}; TACE: {tace_res:.3}')
    return results


def train():
    opt = parse_option()
    if opt.dataset == "cifar10":
        n_cls = 10
    elif opt.dataset == "cifar100":
        n_cls = 100
    elif opt.dataset == "cifar10h":
        n_cls = 10
    elif opt.dataset == 'svhn':
        n_cls = 10

    if opt.semi:
        smi = True
    else:
        smi = False
    if opt.dl:
        dl = True
    else:
        dl = False

    train_loader, val_loader, test_loader, targets = data_loader(opt.dataset, batch_size=128, semi=smi,
                                                                 semi_percent=opt.semi_percent)

    ensemble(opt.ensemble, opt.nh, opt.lamda1, opt.lamda2, opt.dataset, targets, n_cls, test_loader, dl, smi,
             opt.semi_percent,
             opt.model_path,opt.classifier_path)


if __name__ == "__main__":
    train()
