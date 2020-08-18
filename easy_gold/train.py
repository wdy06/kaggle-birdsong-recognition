import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

import datasets
import model_utils
import utils

parser = argparse.ArgumentParser(description="aptos2019 blindness detection on kaggle")
parser.add_argument("--debug", help="run debug mode", action="store_true")
parser.add_argument("--multi", help="use multi gpu", action="store_true")
# parser.add_argument('--config', '-c', type=str, help='path to config')
args = parser.parse_args()

print(f"found {torch.cuda.device_count()} gpus !!")

exp_name = utils.make_experiment_name(args.debug)
RESULT_DIR = utils.RESULTS_BASE_DIR / exp_name
os.mkdir(RESULT_DIR)
print(f"created: {RESULT_DIR}")

device = torch.device("cuda:0")

SAMPLE_RATE = 32000
NUM_WORKERS = 64
BATCH_SIZE = 256
if args.debug:
    EPOCH = 1
    print("running debug mode...")
else:
    EPOCH = 20
classes = pd.read_pickle(utils.DATA_DIR / "classes.pkl")
train_set = pd.read_pickle(utils.DATA_DIR / "train_set.pkl")
val_set = pd.read_pickle(utils.DATA_DIR / "val_set.pkl")

utils.dump_pickle(classes, RESULT_DIR / "classes.pkl")

train_ds = datasets.SpectrogramDataset(
    train_set,
    classes,
    sample_rate=SAMPLE_RATE,
    len_mult=100,
    spec_max=80,
    spec_min=-100,
)
valid_ds = datasets.SpectrogramDataset(
    val_set, classes, sample_rate=SAMPLE_RATE, len_mult=20
)

print(len(train_ds), len(valid_ds))

train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
valid_dl = torch.utils.data.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)


model_name = "base_resnet50"
model = model_utils.build_model(model_name, n_class=len(classes), pretrained=False)
if args.multi:
    print("Using pararell gpu")
    model = nn.DataParallel(model)

# model.cuda()
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), 1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

print("start training...")
for epoch in range(EPOCH):
    t0 = time.time()
    running_loss = 0.0
    for i, data in enumerate(train_dl, 0):
        model.train()
        inputs, labels = data[0].cuda(), data[1].cuda()
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels.argmax(1))
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if i % len(train_dl) == len(train_dl) - 1:
            model.eval()
            preds = []
            targs = []

            with torch.no_grad():
                for data in valid_dl:
                    inputs, labels = data[0].cuda(), data[1].cuda()
                    outputs = model(inputs)
                    preds.append(outputs.cpu().detach())
                    targs.append(labels.cpu().detach())

                preds = torch.cat(preds)
                targs = torch.cat(targs)

            accuracy = (
                (targs.argmax(1) == preds.softmax(-1).argmax(1)).float().mean().item()
            )
            print(
                f"[{epoch + 1}, {time.time() - t0:.1f}] loss: {running_loss / (len(train_dl)-1):.3f}, accuracy: {accuracy:.3f}"
            )
            running_loss = 0.0

model_path = RESULT_DIR / "model.pth"
model_utils.save_pytorch_model(model, model_path)
print(f"save model to {model_path}")
print("finish !!")
