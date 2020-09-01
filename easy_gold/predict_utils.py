import warnings
from collections import OrderedDict
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


def proba_to_label_string(proba, threshold):
    events = proba >= threshold
    all_events = set(np.argwhere(events)[:, 1])
    labels = list(all_events)
    if len(labels) == 0:
        label_string = "nocall"
    else:
        labels_str_list = list(map(lambda x: utils.INV_BIRD_CODE[x], labels))
        label_string = " ".join(labels_str_list)
    # print(label_string)
    return label_string


def prediction_for_clip(
    test_df: pd.DataFrame,
    clip: np.ndarray,
    ds_class,
    sample_rate,
    model,
    composer=None,
    threshold=0.5,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    prediction_dict = {}
    for idx in tqdm(range(len(test_df))):
        record = test_df.loc[idx, :]
        # print(record)
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
            with torch.no_grad():
                prediction = torch.sigmoid(model(image))
                proba = prediction.detach().cpu().numpy()

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

            # all_events = set()
            proba = np.zeros([0, len(utils.BIRD_CODE)])
            for batch_i in range(n_iter):
                batch = image[batch_i * batch_size : (batch_i + 1) * batch_size]
                if batch.ndim == 3:
                    batch = batch.unsqueeze(0)

                batch = batch.to(device)
                with torch.no_grad():
                    prediction = torch.sigmoid(model(batch))
                    _proba = prediction.detach().cpu().numpy()
                # print(proba.shape)
                proba = np.concatenate([proba, _proba])
        # label_string = proba_to_label_string(proba, threshold)
        # prediction_dict[row_id] = label_string
        prediction_dict[row_id] = proba
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
    agg_dict = OrderedDict()
    for model_config in model_list:
        print(model_config)
        model = model_utils.load_pytorch_model(**model_config)
        all_prediction_dict = OrderedDict()
        for audio_id in unique_audio_id:
            clip, _ = librosa.load(
                test_audio / (audio_id + ".mp3"),
                sr=sample_rate,
                mono=True,
                # res_type="kaiser_fast",
            )

            test_df_for_audio_id = test_df.query(
                f"audio_id == '{audio_id}'"
            ).reset_index(drop=True)
            prediction_dict = prediction_for_clip(
                test_df_for_audio_id,
                clip=clip,
                ds_class=ds_class,
                sample_rate=sample_rate,
                model=model,
                composer=composer,
                threshold=threshold,
            )
            all_prediction_dict.update(prediction_dict)

        # aggregate model prediction
        for key in all_prediction_dict.keys():
            if key in agg_dict:
                agg_dict[key] += all_prediction_dict[key]
            else:
                agg_dict[key] = all_prediction_dict[key]

    # print(all_prediction_dict)
    # proba to label string
    for k, v in agg_dict.items():
        v /= len(model_list)
        agg_dict[k] = proba_to_label_string(v, threshold)
    print(agg_dict)
    row_id = list(agg_dict.keys())
    birds = list(agg_dict.values())
    prediction_df = pd.DataFrame({"row_id": row_id, "birds": birds})

    return prediction_df
