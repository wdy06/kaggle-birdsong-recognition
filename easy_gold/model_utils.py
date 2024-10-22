import time
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import f1_score
from torch import nn
from tqdm import tqdm

import timm
from resnest.torch import resnest50_fast_1s1x64d


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
        # x = torch.sigmoid(x)
        return x


def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)


def load_pytorch_model(model_name, path, n_class, in_chans=3, *args, **kwargs):
    state_dict = torch.load(path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if k[:7] == "module.":
            name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model = build_model(model_name, n_class, in_chans, pretrained=False)
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict)
    return model


def build_model(model_name, n_class, in_chans=3, pretrained=False):
    if model_name == "base_resnet50":
        res50 = torchvision.models.resnet50(pretrained=False)
        bottom = nn.Sequential(*list(res50.children())[:6])
        mid = nn.Sequential(*list(res50.children())[6:-2])
        model = nn.Sequential(bottom, mid, Head(n_class))

    elif model_name == "resnest50_fast_1s1x64d":
        if in_chans != 3:
            raise ValueError("resnest50 accepts only 3 channels")
        model = resnest50_fast_1s1x64d(pretrained=pretrained)
        del model.fc
        # # use the same head as the baseline notebook.
        model.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1024, n_class),
        )
    else:
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=n_class, in_chans=in_chans
        )
        # raise ValueError(f"model name {model_name} is not implemented.")

    return model


def predict(model, dataloader, n_class, device, sigmoid=False, tta=1):
    def _predict():
        model.eval()
        model.to(device)
        preds = np.zeros([0, n_class])
        for data in dataloader:
            data = data["image"].to(device)
            with torch.no_grad():
                if sigmoid:
                    y_pred = torch.sigmoid(model(data)).detach()
                else:
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


def train_model(
    epoch,
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    threshold,
    best_model_path,
    logger,
    mixup,
):

    model.to(device)
    best_val_score = 1000000
    for epoch in range(epoch):
        t0 = time.time()
        # training phase
        running_loss = train_1epoch(
            model, train_loader, optimizer, scheduler, criterion, device, mixup
        )

        # validation phase
        if val_loader is not None:
            preds, targs, val_loss = validation(model, val_loader, criterion, device)

            # score = f1_score(preds > threshold, targs, average="micro")
            score = f1_score(preds > threshold, targs > 0, average="micro")
            logger.info(
                f"[{epoch + 1}, {time.time() - t0:.1f}] loss: {running_loss:.4f}, val loss {val_loss:.4f},f1 score: {score:.4f}"
            )
            is_best = bool(val_loss < best_val_score)
            if is_best:
                best_val_score = val_loss
                logger.info(
                    f"update best score !! current best loss: {best_val_score:.5} !!"
                )
                save_pytorch_model(model, best_model_path)
    return best_val_score


def train_1epoch(
    model, data_loader, optimizer, scheduler, criterion, device, mixup=False
):
    running_loss = 0.0
    model.train()
    for data in data_loader:
        inputs, labels = data["image"].to(device), data["targets"].to(device)
        # optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None
        if mixup:
            inputs, targets_a, targets_b, lam = mixup_data(
                inputs, labels, alpha=1.0, device=device
            )
        outputs = model(inputs)
        if mixup:
            loss = mixup_criterion(
                criterion, outputs, targets_a.float(), targets_b.float(), lam
            )
        else:
            loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    running_loss /= len(data_loader)
    return running_loss


def validation(model, data_loader, criterion, device):
    val_loss = 0.0
    model.eval()
    preds = []
    targs = []

    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data["image"].to(device), data["targets"].to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, labels.float())
            preds.append(outputs.cpu().detach().numpy())
            targs.append(labels.cpu().detach().numpy())

        preds = np.concatenate(preds)
        targs = np.concatenate(targs)
        val_loss /= len(data_loader)

    return preds, targs, val_loss


def mixup_data(x, y, alpha=1.0, device=None):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    # if use_cuda:
    #     index = torch.randperm(batch_size).cuda()
    # else:
    #     index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
