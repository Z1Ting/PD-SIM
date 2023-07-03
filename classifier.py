
import torch.nn as nn
import torch
import numpy as np
from models.resnet_simclr_3D import ResNetSimCLR_3D
# from models.resnet_simclr_3D_attention import ResNetSimCLR_3D
import  torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SimCLRClassifier(nn.Module):
    def __init__(self, n_classes, freeze_base, model_pth, hidden_size=512):
        super().__init__()

        model = ResNetSimCLR_3D('resnet18', 128).to(device)  #
        model.load_state_dict(torch.load(model_pth))  #
        self.embeddings = model.features

        if freeze_base:
            print("Freezing embeddings")
            for param in self.embeddings.parameters():
                param.requires_grad = False

        # Only linear projection on top of the embeddings should be enough
        self.classifier = nn.Sequential(nn.Linear(512, 256),
                                        # nn.ReLU(True),
                                        torch.nn.Dropout(p=0.8),
                                        nn.Linear(256, 2),
                                        # nn.Softmax(dim=1),
                                        nn.Sigmoid()
                                        ).to(device)

    def forward(self, X, *args):
        emb = self.embeddings(X)
        emb = emb.flatten(1)
        emb = self.classifier(emb)
        return  emb
