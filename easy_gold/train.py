import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

import datasets
import hydra
import model_utils
import utils

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/config.yml")
def main(cfg):
    logger.info(cfg.pretty())
    logger.info(os.getcwd())
    logger.info(hydra.utils.get_original_cwd())
    utils.seed_everything()

    logger.info(f"found {torch.cuda.device_count()} gpus !!")

    # exp_name = utils.make_experiment_name(cfg.debug)
    # RESULT_DIR = utils.RESULTS_BASE_DIR / exp_name
    # os.mkdir(RESULT_DIR)
    # print(f"created: {RESULT_DIR}")

    device = torch.device("cuda:0")

    df = pd.read_csv(utils.DATA_DIR / "train.csv")
    # remove row becase XC195038.mp3 cannot load
    df = df.drop(df[df.filename == "XC195038.mp3"].index)
    # train_audio_dir = utils.DATA_DIR / "train_audio"
    train_audio_dir = utils.DATA_DIR / "train_resampled_with_nocall"
    nocall_df = pd.read_csv(utils.DATA_DIR / "nocall.csv")

    SAMPLE_RATE = cfg.sample_rate
    NUM_WORKERS = cfg.num_workers
    BATCH_SIZE = cfg.batch_size
    IMAGE_SIZE = cfg.image_size
    # BATCH_SIZE = 10
    if cfg.debug:
        EPOCH = 1
        logger.info(len(df))
        df = df[:1000]
        logger.info("running debug mode...")
    else:
        EPOCH = cfg.epoch

    kfold = StratifiedKFold(n_splits=5)
    for trn_idx, val_idx in kfold.split(df, y=df.ebird_code):
        pass
    logger.info(len(trn_idx))
    logger.info(len(val_idx))

    train_df = df.iloc[trn_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    # concat nocall df
    val_df = pd.concat([val_df, nocall_df]).reset_index()

    composer = utils.build_composer(sample_rate=SAMPLE_RATE, img_size=IMAGE_SIZE)

    train_ds = datasets.SpectrogramDataset(
        train_df, train_audio_dir, sample_rate=SAMPLE_RATE, composer=composer
    )
    valid_ds = datasets.SpectrogramDataset(
        val_df, train_audio_dir, sample_rate=SAMPLE_RATE, composer=composer
    )

    logging.info(f"train_ds: {len(train_ds)}, valid_ds: {len(valid_ds)}")

    train_dl = torch.utils.data.DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    valid_dl = torch.utils.data.DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model_name = "base_resnet50"
    model = model_utils.build_model(
        model_name, n_class=len(utils.BIRD_CODE), pretrained=True
    )
    if cfg.multi:
        logger.info("Using pararell gpu")
        model = nn.DataParallel(model)
        # model = DDP(model)

    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    logger.info("start training...")
    for epoch in range(EPOCH):
        t0 = time.time()
        running_loss = 0.0
        # training phase
        model.train()
        for i, (data) in enumerate(train_dl, 0):
            # print("-" * 10)
            # inputs, labels = data[0].cuda(), data[1].cuda()
            inputs, labels = data["image"].cuda(), data["targets"].cuda()
            optimizer.zero_grad()

            # outputs = model(inputs).double()i

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        # validation phase
        val_loss = 0.0
        # if i % len(train_dl) == len(train_dl) - 1:
        model.eval()
        preds = []
        targs = []

        with torch.no_grad():
            for data in valid_dl:
                inputs, labels = data["image"].cuda(), data["targets"].cuda()
                # inputs = data.to(device)
                # outputs = model(inputs).double()
                outputs = model(inputs)
                val_loss += criterion(outputs, labels.float())
                preds.append(outputs.cpu().detach().numpy())
                targs.append(labels.cpu().detach().numpy())

            preds = np.concatenate(preds)
            targs = np.concatenate(targs)
            val_loss /= len(valid_dl)
            # val_loss = criterion(preds, targs)

        threshold = cfg.threshold

        score = f1_score(preds > threshold, targs, average="micro")
        logger.info(
            f"[{epoch + 1}, {time.time() - t0:.1f}] loss: {running_loss / (len(train_dl)-1):.4f}, val loss {val_loss:.4f},f1 score: {score:.4f}"
        )
        running_loss = 0.0

    model_path = "model.pth"
    model_utils.save_pytorch_model(model, model_path)
    logger.info(f"save model to {model_path}")
    logger.info("finish !!")


if __name__ == "__main__":
    main()
