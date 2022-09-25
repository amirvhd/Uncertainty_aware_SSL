import logging
import os.path as osp
from models.resnet_big import SupConResNet, LinearClassifier
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from utils.score_utils import calc_metrics_transformed
from models.concatenate import MyEnsemble
# from laplace import Laplace
from Dataloader.dataloader import data_loader
from Train.linear_eval import set_model_linear

logger = logging.getLogger(__name__)


def set_model(model_name, n_cls, path):
    model = SupConResNet(name=model_name)
    criterion = torch.nn.CrossEntropyLoss()

    classifier = LinearClassifier(name=model_name, num_classes=n_cls)

    state_dict = torch.load(path)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        model.load_state_dict(state_dict)

    return model, classifier, criterion


class LitBaseline(pl.LightningModule):
    def __init__(self, ind_name, linear_path, model_path, n_cls, out_dir, nh, LA=False):
        super().__init__()
        # self.model_name = model_name
        self.ind_name = ind_name
        self.LA = LA
        self.nh = nh
        model, classifier, criterion = set_model_linear("resnet50", n_cls, model_path, nh=self.nh)
        classifier.load_state_dict(torch.load(linear_path))

        if LA:
            linear_model_map = MyEnsemble(model.encoder, classifier).cuda().eval()
            train_loader, val_loader, _, _ = data_loader()
            la = Laplace(linear_model_map, 'classification',
                         subset_of_weights='last_layer',
                         hessian_structure='kron')

            la.fit(train_loader)
            la.optimize_prior_precision(method='CV', val_loader=val_loader)
            self.model1 = la

        self.model = model
        self.classifier = classifier
        # IND
        self.is_ind = True
        self.ind_max_probs = None
        self.avg_train_norm = 0

        # OOD
        self.ood_name = ""

        # Metrics
        self.baseline_res = None
        self.is_save_scores = False

        self.validation_size = 0
        self.out_dir = out_dir

        self.pinv_rcond = 1e-15  # default

    def set_validation_size(self, validation_size: int):
        self.validation_size = validation_size
        logger.info(f"set_validation_size: validation_size={self.validation_size}")

    def set_ood(self, ood_name: str):
        self.ood_name = ood_name
        self.is_ind = ood_name == self.ind_name
        logger.info(f"set_ood: ood_name={self.ood_name} is_ind={self.is_ind}")

    def configure_optimizers(self):
        # We don't want to train, set lr to 0 and gives optimizer different param to optimize on
        return torch.optim.Adam(self.model.parameters(), lr=0.0)

    def backward(self, loss, optimizer, optimizer_idx, *args, **kwargs):
        pass

    def forward(self, x):
        if self.LA:
            # self.model1.eval()
            with torch.no_grad():
                logits = self.model1(x)
        else:
            self.model.eval()
            self.classifier.eval()
            with torch.no_grad():
                features = self.model.encoder(x)
                logits = self.classifier(features)

        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.LA:
            probs = self.forward(x)
            logits = 0
            loss = F.cross_entropy(probs, y)
        else:
            logits = self.forward(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            loss = F.cross_entropy(logits, y)

        y_hat = probs.argmax(dim=-1)
        is_correct = y_hat == y
        output = {
            "probs": probs,
            "loss": loss,
            "is_correct": is_correct,
            "logits": logits,
        }
        return output

    def training_epoch_end(self, outputs):
        acc = torch.hstack([out["is_correct"] for out in outputs]).float().mean() * 100
        print("\nTraining set acc {:.2f}%".format(acc))

    def test_step(self, batch, batch_idx):
        x, y = batch
        if self.LA:
            probs = self.forward(x)
            logits = 0
        else:
            logits = self.forward(x)
            probs = torch.nn.functional.softmax(logits, dim=-1)
        y_hat = probs.argmax(dim=-1)
        is_correct = y_hat == y
        output = {
            "probs": probs,
            "is_correct": is_correct,
            "logits": logits,
            "target": y
        }
        return output

    def test_epoch_end(self, outputs):
        probs = torch.vstack([out["probs"] for out in outputs])
        logits = torch.vstack([out["logits"] for out in outputs])
        acc = torch.hstack([out["is_correct"] for out in outputs]).float().mean() * 100
        # Compute the normalization factor
        max_probs = torch.max(probs, dim=-1).values.cpu().numpy()
        targets = torch.cat([out["target"] for out in outputs])
        if self.is_ind:
            print("\nValidation set acc {:.2f}%".format(acc))
            # Store IND scores
            self.ind_max_probs = max_probs
        else:
            # Run evaluation on the OOD set
            self.baseline_res = calc_metrics_transformed(self.ind_max_probs, max_probs)
        if self.is_save_scores is True:
            np.savetxt(
                osp.join(self.out_dir, f"{self.ood_name}_baseline.txt"), max_probs
            )

    def get_performance(self):
        return self.baseline_res
