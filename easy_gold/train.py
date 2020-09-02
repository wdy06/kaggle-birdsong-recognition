import argparse
import logging
import os
import time
from logging import log

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
from runner import Runner

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs/config.yml")
def main(cfg):
    logger.info(cfg.pretty())
    logger.info(os.getcwd())
    logger.info(hydra.utils.get_original_cwd())
    utils.seed_everything()

    logger.info(f"found {torch.cuda.device_count()} gpus !!")

    if cfg.gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    df = pd.read_csv(utils.DATA_DIR / "train.csv")
    # remove row becase XC195038.mp3 cannot load
    df = df.drop(df[df.filename == "XC195038.mp3"].index)
    # train_audio_dir = utils.DATA_DIR / "train_audio"
    train_audio_dir = utils.DATA_DIR / "train_resampled_with_nocall"
    nocall_df = pd.read_csv(utils.DATA_DIR / "nocall.csv")

    SAMPLE_RATE = cfg.sample_rate
    IMAGE_SIZE = cfg.image_size

    n_class = len(utils.BIRD_CODE)
    # best_model_path = "best_model.pth"

    if cfg.debug:
        logger.info(len(df))
        df = df[:1000]
        logger.info("running debug mode...")
    else:
        EPOCH = cfg.epoch

    fold_indices = []
    kfold = StratifiedKFold(n_splits=5)
    for trn_idx, val_idx in kfold.split(df, y=df.ebird_code):
        fold_indices.append((trn_idx, val_idx))

    if cfg.debug:
        fold_indices = [fold_indices[0]]

    composer = utils.build_composer(
        sample_rate=SAMPLE_RATE,
        img_size=IMAGE_SIZE,
        in_chans=cfg.model.in_chans,
        melspectrogram_parameters=cfg.composer.melspectrogram_parameters,
    )

    runner = Runner(
        df=df,
        epoch=cfg.epoch,
        config=cfg,
        n_class=n_class,
        composer=composer,
        data_dir=train_audio_dir,
        save_dir=".",
        logger=logger,
        device=device,
        fold_indices=fold_indices,
    )

    oof_preds, avg_val_loss = runner.run_train_cv()
    print(oof_preds)
    runner.run_train_all()
    preds_nocall = runner.run_predict_cv(nocall_df)
    oof_preds = np.concatenate([oof_preds, preds_nocall], axis=0)

    # optimize threshold
    rounder = utils.OptimizeRounder(n_class)
    df = pd.concat([df, nocall_df], axis=0)
    y_true = np.array(df.ebird_code.map(lambda x: utils.one_hot_encode(x)).tolist())
    # rounder.fit(y_true, oof_preds)
    # utils.dump_json(rounder.coefficients(), "threshold.json")
    # best_val_score = f1_score(
    #     y_true, oof_preds > rounder.coefficients(), average="micro"
    # )
    best_val_score = f1_score(y_true, oof_preds > cfg.threshold, average="micro")

    utils.dump_pickle(oof_preds, "oof_preds.pkl")
    utils.dump_pickle(y_true, "oof_true.pkl")
    # logger.info(f"best f1_score: {best_val_score}")
    logger.info(f"threshold {cfg.threshold} f1_score: {best_val_score}")
    logger.info(f"average best loss: {avg_val_loss}")

    # print(preds)
    logger.info(os.getcwd())
    logger.info("finish !!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(e)
