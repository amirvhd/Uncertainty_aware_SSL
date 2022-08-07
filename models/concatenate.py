import torch.nn as nn




class MyEnsemble(nn.Module):
    def __init__(self, modelA,classifier ):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.classifier = classifier

    def forward(self, x):
        x1 = self.modelA(x)
        x = self.classifier(x1)
        return x