import logging
import time
import pandas as pd
from utils.baseline_lit_utils import LitBaseline
logger = logging.getLogger(__name__)


def execute_baseline(
        opt, lit_model_h: LitBaseline, trainer, loaders_dict
):
    # Eval training set (create (x^T x)^{-1} matrix)
    train_loader = loaders_dict.pop("trainset")
    trainer.fit(lit_model_h, train_dataloaders=train_loader)

    # Get in-distribution scores
    ind_loader = loaders_dict.pop(opt.dataset)
    trainer.test(lit_model_h, dataloaders=ind_loader, verbose=False)

    # Eval out-of-distribution datasets
    baseline_results = []
    for i, (ood_name, ood_dataloader) in enumerate(loaders_dict.items()):
        t0 = time.time()
        # Evaluate ood scores
        lit_model_h.set_ood(ood_name)
        trainer.test(lit_model_h, dataloaders=ood_dataloader, ckpt_path=None, verbose=False)
        baseline_res = lit_model_h.get_performance()

        baseline_res["ood_name"] = ood_name

        # Save
        baseline_results.append(baseline_res)
    return pd.DataFrame(baseline_results)
