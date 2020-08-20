from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torchvision
from torch import nn
from tqdm import tqdm


class Head(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.layers = nn.ModuleList(
            [
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(1024, self.n_class),
            ]
        )

    def forward(self, x):
        #         set_trace()
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        for layer in self.layers:
            x = layer(x)
        x = torch.sigmoid(x)
        return x


def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)


def load_pytorch_model(model_name, path, n_class, *args, **kwargs):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == "module.":
            name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model = build_model(model_name, n_class, pretrained=False)
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict)
    return model


def build_model(model_name, n_class, pretrained=False):
    if model_name == "base_resnet50":
        res50 = torchvision.models.resnet50(pretrained=False)
        bottom = nn.Sequential(*list(res50.children())[:6])
        mid = nn.Sequential(*list(res50.children())[6:-2])
        model = nn.Sequential(bottom, mid, Head(n_class))
    else:
        raise ValueError(f"model name {model_name} is not implemented.")

    return model


def predict(model, dataloader, n_class, device, tta=1):
    def _predict():
        model.eval()
        model.to(device)
        preds = np.zeros([0, n_class])
        for data, _ in dataloader:
            data = data.to(device)
            with torch.no_grad():
                y_pred = model(data).detach()
            # y_pred = F.softmax(y_pred, dim=1).cpu().numpy()
            y_pred = y_pred.cpu().numpy()
            preds = np.concatenate([preds, y_pred])
        return preds

    if tta > 1:
        print("use tta ...")
    preds = 0
    for i in tqdm(range(tta)):
        preds += _predict()
    preds /= tta
    return preds


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))
