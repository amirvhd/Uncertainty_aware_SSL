import torch
import argparse
from torchvision.models import resnet50
from laplace import Laplace
from netcal.metrics import ECE
import torch.distributions as dists
from models.concatenate import MyEnsemble
import numpy as np
from Dataloader.dataloader import data_loader
from Train.linear_eval import set_model_linear, predict
from utils.metrics import OELoss, SCELoss, TACELoss, ACELoss


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'isic'], help='dataset')
    parser.add_argument('--model_path', type=str,
                        default='./saved_models/{}_models_UAloss/simclr800_linear_{}_epoch100.pt', help='model path')
    parser.add_argument('--semi', action='store_true',
                        help='semi-supervised')
    parser.add_argument('--semi_percent', type=int, default=10,
                        help='percentage of data usage in semi-supervised')
    parser.add_argument('--nh', type=int, default=1,
                        help='number of heads')
    opt = parser.parse_args()
    return opt


def ensemble(n, nh, dataset, targets, n_cls, test_loader, semi=False, semi_percent=1,
             path='./saved_models/{}_models_UAloss/simclr800_linear_{}_epoch100.pt'):
    probs_ensemble2_model = []
    if semi == True:
        for i in range(n):
            linear_model_path = './saved_models/{}_models/simclr_linear_{}_epoch40_percent{}.pt'.format(dataset, i,
                                                                                                        semi_percent)
            simclr_path = './saved_models/{}_models/simclr_encoder_{}_epoch40_percent{}.pt'.format(dataset, i,
                                                                                                   semi_percent)
            model, classifier, criterion = set_model_linear("resnet50", n_cls, simclr_path)
            classifier.load_state_dict(torch.load(linear_model_path))
            linear_model = MyEnsemble(model.encoder, classifier).cuda().eval()
            probs_ensemble2_model.append(predict(test_loader, linear_model, laplace=False))
        probs_ensemble2_model = np.array(probs_ensemble2_model)
        probs_ensemble2 = np.mean(probs_ensemble2_model, 0)
        acc_ensemble2 = (probs_ensemble2.argmax(-1) == targets).mean()
        ece_ensemble2 = ECE(bins=15).measure(probs_ensemble2, targets)
        nll_ensemble2 = -dists.Categorical(torch.tensor(probs_ensemble2)).log_prob(torch.tensor(targets)).mean()
        print("number of ensemble is {}".format(n))
        print(f'[ensemble] Acc.: {acc_ensemble2:.1%}; ECE: {ece_ensemble2:.1%}; NLL: {nll_ensemble2:.3}')
    else:
        for i in range(n):
            linear_model_path = './saved_models/{}_models_UAloss/simclr800_linear_{}_epoch100_{}heads.pt'.format(
                dataset,
                i, nh)
            simclr_path = './saved_models/{}_models_UAloss/simclr800_encoder_{}_epoch100_{}heads.pt'.format(dataset, i,
                                                                                                            nh)
            model, classifier, criterion = set_model_linear("resnet50", n_cls, simclr_path)
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
        oe_res = oe.loss(output=probs_ensemble2, labels=targets, logits=False)
        sce_res = sce.loss(output=probs_ensemble2, labels=targets, logits=False)
        ace_res = ace.loss(output=probs_ensemble2, labels=targets, logits=False)
        tace_res = tace.loss(output=probs_ensemble2, labels=targets, logits=False)
        ece_ensemble2 = ECE(bins=15).measure(probs_ensemble2, targets)
        nll_ensemble2 = -dists.Categorical(torch.tensor(probs_ensemble2)).log_prob(torch.tensor(targets)).mean()
        print("number of ensemble is {}".format(n))
        print(
            f'[ensemble] Acc.: {acc_ensemble2:.1%}; ECE: {ece_ensemble2:.1%}; NLL: {nll_ensemble2:.3}; '
            f'OE: {oe_res:.3}; SCE: {sce_res:.3}; ACE: {ace_res:.3}; TACE: {tace_res:.3}')


def ensemble_laplace(n, dataset, targets, n_cls, test_loader, train_loader, val_loader, semi, semi_percent):
    probs_ensemble2_model = []
    if semi == True:
        for i in range(n):
            linear_model_path = './saved_models/{}_models/simclr_linear_{}_epoch40_percent{}.pt'.format(dataset, i,
                                                                                                        semi_percent)
            simclr_path = './saved_models/{}_models/simclr_encoder_{}_epoch40_percent{}.pt'.format(dataset, i,
                                                                                                   semi_percent)
            model, classifier, criterion = set_model_linear("resnet50", n_cls, simclr_path)
            classifier.load_state_dict(torch.load(linear_model_path))
            linear_model = MyEnsemble(model.encoder, classifier).cuda().eval()
            la = Laplace(linear_model, 'classification',
                         subset_of_weights='last_layer',
                         hessian_structure='kron')
            la.fit(train_loader)
            la.optimize_prior_precision(method='CV', val_loader=val_loader)
            probs_ensemble2_model.append(predict(test_loader, la, laplace=True))
        probs_ensemble2_model = np.array(probs_ensemble2_model)
        probs_ensemble2 = np.mean(probs_ensemble2_model, 0)
        acc_ensemble2 = (probs_ensemble2.argmax(-1) == targets).mean()
        ece_ensemble2 = ECE(bins=15).measure(probs_ensemble2, targets)
        nll_ensemble2 = -dists.Categorical(torch.tensor(probs_ensemble2)).log_prob(torch.tensor(targets)).mean()
        print("number of ensemble is {}".format(n))
        print(f'[ensemble_laplace] Acc.: {acc_ensemble2:.1%}; ECE: {ece_ensemble2:.1%}; NLL: {nll_ensemble2:.3}')
    else:
        for i in range(n):
            linear_model_path = './saved_models/{}_models/simclr_linear_{}_epoch40.pt'.format(dataset, i)
            simclr_path = './saved_models/{}_models/simclr_encoder_{}_epoch40.pt'.format(dataset, i)
            model, classifier, criterion = set_model_linearl("resnet50", n_cls, simclr_path)
            classifier.load_state_dict(torch.load(linear_model_path))
            linear_model = MyEnsemble(model.encoder, classifier).cuda().eval()
            la = Laplace(linear_model, 'classification',
                         subset_of_weights='last_layer',
                         hessian_structure='kron')
            la.fit(train_loader)
            la.optimize_prior_precision(method='CV', val_loader=val_loader)
            probs_ensemble2_model.append(predict(test_loader, la, laplace=True))
        probs_ensemble2_model = np.array(probs_ensemble2_model)
        probs_ensemble2 = np.mean(probs_ensemble2_model, 0)
        acc_ensemble2 = (probs_ensemble2.argmax(-1) == targets).mean()
        ece_ensemble2 = ECE(bins=15).measure(probs_ensemble2, targets)
        nll_ensemble2 = -dists.Categorical(torch.tensor(probs_ensemble2)).log_prob(torch.tensor(targets)).mean()
        print("number of ensemble is {}".format(n))
        print(f'[ensemble_laplace] Acc.: {acc_ensemble2:.1%}; ECE: {ece_ensemble2:.1%}; NLL: {nll_ensemble2:.3}')


def train():
    opt = parse_option()
    if opt.dataset == "cifar10":
        n_cls = 10
    elif opt.dataset == "cifar100":
        n_cls = 100
    elif opt.dataset == 'svhn':
        n_cls = 10
    elif opt.dataset == 'isic':
        n_cls = 7
    if opt.semi:
        smi = True
    else:
        smi = False
    train_loader, val_loader, test_loader, targets = data_loader(opt.dataset, batch_size=512, semi=smi,
                                                                 semi_percent=opt.semi_percent)
    # MAP calculation for random seed = 1
    # la.optimize_prior_precision(method='CV', val_loader=val_loader, link_approx="mc", n_samples=200, pred_type="glm")
    # la.optimize_prior_precision(method='marglik', link_approx="mc", n_samples=200)
    # la.optimize_prior_precision(pred_type="nn", method='marglik', link_approx="mc", n_samples=200)
    # la.optimize_prior_precision(pred_type="nn", method='CV', val_loader=val_loader, link_approx="mc", n_samples=1000)
    # MAP result for first case

    ensemble(1, opt.nh, opt.dataset, targets, n_cls, test_loader, smi, opt.semi_percent, opt.model_path)

    # Laplace result for first case

    # ensemble_laplace(1, opt.dataset, targets, n_cls, test_loader, train_loader, val_loader, smi, opt.semi_percent)

    # Ensemble for 2 different random seed
    n = 2
    # ensemble(n, opt.dataset, targets, n_cls, test_loader, smi, opt.semi_percent)

    # Ensemble for 10 different random seed
    n = 10
    # ensemble(n, opt.dataset, targets, n_cls, test_loader, smi, opt.semi_percent)

    # Ensemble for 2 different random seed for laplace
    n = 2
    # ensemble_laplace(n, opt.dataset, targets, n_cls, test_loader, train_loader, val_loader, smi, opt.semi_percent)

    # Ensemble for 10 different random seed for laplace
    n = 10
    # ensemble_laplace(n, opt.dataset, targets, n_cls, test_loader, train_loader, val_loader, smi, opt.semi_percent)


if __name__ == "__main__":
    train()
