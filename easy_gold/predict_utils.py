import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import librosa
import model_utils
import utils


def long_clip_to_images(y, sample_rate, composer):
    len_y = len(y)
    start = 0
    end = sample_rate * 5
    images = []
    while len_y > start:
        y_batch = y[start:end].astype(np.float32)
        if len(y_batch) != (sample_rate * 5):
            break
        start = end
        end = end + sample_rate * 5

        image = composer(y_batch)
        images.append(image)

    return images


def prediction_for_clip(
    test_df: pd.DataFrame,
    clip: np.ndarray,
    ds_class,
    sample_rate,
    model_list,
    composer=None,
    threshold=0.5,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model.to(device)

    # model.eval()
    prediction_dict = {}
    for idx in tqdm(range(len(test_df))):
        record = test_df.loc[idx, :]
        print(record)
        row_id = record.row_id
        site = record.site
        if site in {"site_1", "site_2"}:
            end_seconds = int(record.seconds)
            start_seconds = int(end_seconds - 5)

            start_index = sample_rate * start_seconds
            end_index = sample_rate * end_seconds
            y = clip[start_index:end_index].astype(np.float32)
            image = composer(y)
            image = image[np.newaxis, :, :, :]
            image = torch.Tensor(image)
            image = image.to(device)
            proba = 0
            for config in tqdm(model_list):
                model = model_utils.load_pytorch_model(**config)
                model.to(device)
                model.eval()
                with torch.no_grad():
                    prediction = model(image)
                    #                 proba = prediction["multilabel_proba"].detach().cpu().numpy().reshape(-1)
                    proba += prediction.detach().cpu().numpy().reshape(-1)
            proba /= len(model_list)

            events = proba >= threshold
            labels = np.argwhere(events).reshape(-1).tolist()

        else:
            # to avoid prediction on large batch
            y = clip.astype(np.float32)
            images = long_clip_to_images(y, sample_rate, composer)
            image = np.asarray(images)
            image = torch.Tensor(image)
            image = image.to(device)
            image = image.squeeze(0)
            batch_size = 16
            whole_size = image.size(0)
            if whole_size % batch_size == 0:
                n_iter = whole_size // batch_size
            else:
                n_iter = whole_size // batch_size + 1

            all_events = set()
            all_proba = np.zeros([0, len(utils.BIRD_CODE)])
            for batch_i in range(n_iter):
                batch = image[batch_i * batch_size : (batch_i + 1) * batch_size]
                if batch.ndim == 3:
                    batch = batch.unsqueeze(0)

                batch = batch.to(device)
                proba = 0
                for config in tqdm(model_list):
                    model = model_utils.load_pytorch_model(**config)
                    model.to(device)
                    model.eval()
                    with torch.no_grad():
                        prediction = model(batch)
                        proba += prediction.detach().cpu().numpy()
                proba /= len(model_list)
                print(proba.shape)
                all_proba = np.concatenate([all_proba, proba])

            # events = proba >= threshold
            events = all_proba >= threshold
            for i in range(len(events)):
                event = events[i, :]
                labels = np.argwhere(event).reshape(-1).tolist()
                for label in labels:
                    all_events.add(label)
            print(all_proba.shape)
            labels = list(all_events)
            print(labels)
        if len(labels) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            #             labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))
            labels_str_list = list(map(lambda x: utils.INV_BIRD_CODE[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string
    return prediction_dict


def prediction(
    test_df: pd.DataFrame,
    test_audio: Path,
    ds_class,
    model_list,
    composer=None,
    sample_rate=32000,
    threshold=0.5,
):
    unique_audio_id = test_df.audio_id.unique()

    warnings.filterwarnings("ignore")
    prediction_dfs = []
    for audio_id in unique_audio_id:
        clip, _ = librosa.load(
            test_audio / (audio_id + ".mp3"),
            sr=sample_rate,
            mono=True,
            # res_type="kaiser_fast",
        )

        test_df_for_audio_id = test_df.query(f"audio_id == '{audio_id}'").reset_index(
            drop=True
        )
        prediction_dict = prediction_for_clip(
            test_df_for_audio_id,
            clip=clip,
            ds_class=ds_class,
            sample_rate=sample_rate,
            model_list=model_list,
            composer=composer,
            threshold=threshold,
        )
        row_id = list(prediction_dict.keys())
        birds = list(prediction_dict.values())
        prediction_df = pd.DataFrame({"row_id": row_id, "birds": birds})
        prediction_dfs.append(prediction_df)

    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df
