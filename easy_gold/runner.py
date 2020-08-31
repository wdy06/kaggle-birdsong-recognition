from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch import nn

import datasets
import model_utils


class Runner:
    def __init__(
        self,
        df,
        # nocall_df,
        epoch,
        config,
        n_class,
        composer,
        data_dir,
        save_dir,
        logger,
        device,
        fold_indices=None,
    ):
        super().__init__()
        self.df = df
        # self.nocall_df = nocall_df
        self.epoch = epoch
        self.config = config
        self.n_class = n_class
        self.composer = composer
        self.data_dir = data_dir
        self.save_dir = Path(save_dir)
        self.logger = logger
        self.device = device
        self.fold_indices = fold_indices

    # def train(self, train_x, train_y, val_x=None, val_y=None):
    #     pass
    # validation = True
    # if (val_x is None) or (val_y is None):
    #     validation = False
    # model = self.build_model()
    # if validation:
    #     model.fit(train_x, train_y, val_x, val_y)
    #     y_pred = model.predict(val_x)
    #     return model, y_pred
    # else:
    #     model.fit(train_x, train_y)
    #     return model, None

    def run_train_cv(self):
        oof_preds = np.zeros((len(self.df), self.n_class))
        for i_fold, (trn_idx, val_idx) in enumerate(self.fold_indices):
            self.logger.info("-" * 10)
            self.logger.info(f"fold: {i_fold}")
            train_df = self.df.iloc[trn_idx].reset_index(drop=True)
            val_df = self.df.iloc[val_idx].reset_index(drop=True)
            # concat nocall df
            # val_df = pd.concat([val_df, self.nocall_df]).reset_index()
            train_ds = datasets.SpectrogramDataset(
                train_df,
                self.data_dir,
                sample_rate=self.config.sample_rate,
                composer=self.composer,
            )
            valid_ds = datasets.SpectrogramDataset(
                val_df,
                self.data_dir,
                sample_rate=self.config.sample_rate,
                composer=self.composer,
            )
            train_dl = torch.utils.data.DataLoader(
                train_ds, shuffle=True, **self.config.dataloader
            )
            valid_dl = torch.utils.data.DataLoader(
                valid_ds, shuffle=False, **self.config.dataloader
            )
            model = model_utils.build_model(
                self.config.model.name,
                n_class=self.n_class,
                pretrained=self.config.model.pretrained,
            )
            if self.config.multi:
                self.logger.info("Using pararell gpu")
                model = nn.DataParallel(model)

            # criterion = nn.BCELoss()
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), float(self.config.learning_rate))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
            best_model_path = self.save_dir / f"best_model_fold{i_fold}.pth"
            model_utils.train_model(
                epoch=self.epoch,
                model=model,
                train_loader=train_dl,
                val_loader=valid_dl,
                optimizer=optimizer,
                scheduler=scheduler,
                criterion=criterion,
                device=self.device,
                threshold=self.config.threshold,
                best_model_path=best_model_path,
                logger=self.logger,
            )
            model = model_utils.load_pytorch_model(
                model_name=self.config.model.name,
                path=best_model_path,
                n_class=self.n_class,
            )
            preds = model_utils.predict(
                model, valid_dl, self.n_class, self.device, sigmoid=True
            )
            oof_preds[val_idx, :] = preds
        # oof_score = self.metrics(self.y, oof_preds)
        return oof_preds

    def run_predict_cv(self, df):
        ds = datasets.SpectrogramDataset(
            df,
            self.data_dir,
            sample_rate=self.config.sample_rate,
            composer=self.composer,
        )
        dataloader = torch.utils.data.DataLoader(
            ds, shuffle=False, **self.config.dataloader
        )
        preds = np.zeros((len(df), self.n_class))
        for i_fold, _ in enumerate(self.fold_indices):
            model_path = self.save_dir / f"best_model_fold{i_fold}.pth"
            model = model_utils.load_pytorch_model(
                model_name=self.config.model.name, path=model_path, n_class=self.n_class
            )
            preds += model_utils.predict(
                model, dataloader, self.n_class, self.device, sigmoid=True
            )
        preds /= len(self.fold_indices)
        return preds

    def run_train_all(self):
        self.logger.info(f"training on all data...")
        train_ds = datasets.SpectrogramDataset(
            self.df,
            self.data_dir,
            sample_rate=self.config.sample_rate,
            composer=self.composer,
        )
        train_dl = torch.utils.data.DataLoader(
            train_ds, shuffle=True, **self.config.dataloader
        )
        model = model_utils.build_model(
            self.config.model.name,
            n_class=self.n_class,
            pretrained=self.config.model.pretrained,
        )
        if self.config.multi:
            self.logger.info("Using pararell gpu")
            model = nn.DataParallel(model)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), float(self.config.learning_rate))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
        model_utils.train_model(
            epoch=self.epoch,
            model=model,
            train_loader=train_dl,
            val_loader=None,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=self.device,
            threshold=self.config.threshold,
            best_model_path=None,
            logger=self.logger,
        )
        model_utils.save_pytorch_model(model, self.save_dir / "all_model.pth")
        self.logger.info(f'save model to {self.save_dir / "all_model.pth"}')

    def run_predict_all(self, test_x):
        pass
        # model = self.build_model()
        # model.load_model(self.save_dir / f"{self.run_name}_all.pkl")
        # preds = model.predict(test_x)
        # return preds

    def get_oof_preds(self):
        pass
        # oof_preds = np.zeros(len(self.y))
        # for i_fold, (trn_idx, val_idx) in enumerate(self.fold_indeices):
        #     val_x = self.x.iloc[val_idx]
        #     model = self.build_model()
        #     model.load_model(self.save_dir / f"{self.run_name}_fold{i_fold}.pkl")
        #     oof_preds[val_idx] = model.predict(val_x)
        # return oof_preds, self.y

    def save_importance_cv(self):
        raise NotImplementedError
