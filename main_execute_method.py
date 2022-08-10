import time
import argparse

import pandas as pd
import pytorch_lightning as pl
import torch

from Dataloader.dataset_utils import get_data_loaders
from utils.baseline_lit_utils import LitBaseline
from utils.method_utils import execute_baseline


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn', 'isic'], help='dataset')
    parser.add_argument('--data_dir', type=str, default='../../DATA/',
                        help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--dev_run', action='store_true',
                        help='')
    parser.add_argument('--LA', action='store_true',
                        help='')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='number of ensemble models')
    opt = parser.parse_args()
    return opt


def run_experiment():

    t0 = time.time()
    opt = parse_option()
    # out_dir = os.getcwd()
    out_dir = "./logs"
    pl.seed_everything(opt.seed)
    if opt.dev_run:
        dev_run = True
    else:
        dev_run = False

    # Initialize trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        fast_dev_run=dev_run,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
        max_epochs=1,
        amp_backend="apex",
        default_root_dir=out_dir,
    )
    baseline_df = pd.DataFrame()
    if opt.dataset == 'cifar10':
        n_cls = 10
    elif opt.dataset == 'cifar100':
        n_cls = 100
    elif opt.dataset == 'svhn':
        n_cls = 10
    elif opt.dataset == 'isic':
        n_cls = 7
    # Execute method
    for i in range(opt.ensemble):
        print("ensmeble number is {}".format(i))
        # Load datasets
        loaders_dict = get_data_loaders(
            opt.dataset,
            opt.data_dir,
            opt.batch_size,
            opt.num_workers if dev_run is False else 0,
            dev_run,
        )
        if opt.LA:
            LA = True
        else:
            LA = False

        linear_model_path = "./saved_models/{}_models_UAloss/simclr800_linear_{}_epoch100_5heads.pt".format(opt.dataset, i)
        simclr_path = "./saved_models/{}_models_UAloss/simclr800_encoder_{}_epoch100_5heads.pt".format(opt.dataset, i)
        lit_model_h = LitBaseline(opt.dataset, linear_model_path, simclr_path, n_cls, out_dir, LA)
        baseline_df = baseline_df.append(execute_baseline(opt, lit_model_h, trainer, loaders_dict))
    ens = baseline_df.groupby(['ood_name']).mean()
    print(ens)
    print("Finish in {:.2f} sec. out_dir={}".format(time.time() - t0, out_dir))


if __name__ == "__main__":
    run_experiment()
