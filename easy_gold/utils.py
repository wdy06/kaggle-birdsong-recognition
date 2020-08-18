import json
import os
import pickle
import random
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

import librosa

# ON_KAGGLE: bool = 'KAGGLE_WORKING_DIR' in os.environ
ON_KAGGLE: bool = "KAGGLE_URL_BASE" in os.environ
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
CONFIG_DIR = BASE_DIR / "configs"
DATA_DIR = Path("../input/birdsong-recognition/") if ON_KAGGLE else BASE_DIR / "data"
FEATURE_DIR = BASE_DIR / "features"
RESULTS_BASE_DIR = Path(".") if ON_KAGGLE else BASE_DIR / "results"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def make_experiment_name(debug):
    experiment_name = datetime.strftime(datetime.now(), "%Y%m%d%H%M%S")
    if debug:
        experiment_name = "debug-" + experiment_name
    return experiment_name


def dump_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4, cls=NpEncoder)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def dump_yaml(obj, path):
    with open(path, "w") as f:
        yaml.dump(obj, f, default_flow_style=False)


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def prediction_for_clip(
    test_df: pd.DataFrame,
    clip: np.ndarray,
    ds_class,
    model,
    classes,
    sample_rate,
    threshold=0.5,
):

    dataset = ds_class(
        df=test_df, clip=clip, sample_rate=sample_rate, spec_min=-100, spec_max=80
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    model.eval()
    prediction_dict = {}
    for image, row_id, site in tqdm(loader):
        #         print(row_id, site)
        #         print(image.shape)
        site = site[0]
        row_id = row_id[0]
        if site in {"site_1", "site_2"}:
            image = image.to(device)

            with torch.no_grad():
                prediction = model(image)
                #                 proba = prediction["multilabel_proba"].detach().cpu().numpy().reshape(-1)
                proba = prediction.detach().cpu().numpy().reshape(-1)

            events = proba >= threshold
            labels = np.argwhere(events).reshape(-1).tolist()

        else:
            # to avoid prediction on large batch
            image = image.squeeze(0)
            batch_size = 16
            whole_size = image.size(0)
            if whole_size % batch_size == 0:
                n_iter = whole_size // batch_size
            else:
                n_iter = whole_size // batch_size + 1

            all_events = set()
            for batch_i in range(n_iter):
                batch = image[batch_i * batch_size : (batch_i + 1) * batch_size]
                if batch.ndim == 3:
                    batch = batch.unsqueeze(0)

                batch = batch.to(device)
                with torch.no_grad():
                    prediction = model(batch)
                    #                     proba = prediction["multilabel_proba"].detach().cpu().numpy()
                    proba = prediction.detach().cpu().numpy()

                events = proba >= threshold
                for i in range(len(events)):
                    event = events[i, :]
                    labels = np.argwhere(event).reshape(-1).tolist()
                    for label in labels:
                        all_events.add(label)

            labels = list(all_events)
        #         print(labels)
        if len(labels) == 0:
            prediction_dict[row_id] = "nocall"
        else:
            #             labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))
            labels_str_list = list(map(lambda x: classes[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string
    return prediction_dict


def prediction(
    test_df: pd.DataFrame,
    test_audio: Path,
    ds_class,
    model,
    classes,
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
            res_type="kaiser_fast",
        )

        test_df_for_audio_id = test_df.query(f"audio_id == '{audio_id}'").reset_index(
            drop=True
        )
        prediction_dict = prediction_for_clip(
            test_df_for_audio_id,
            clip=clip,
            ds_class=ds_class,
            model=model,
            classes=classes,
            sample_rate=sample_rate,
            threshold=threshold,
        )
        row_id = list(prediction_dict.keys())
        birds = list(prediction_dict.values())
        prediction_df = pd.DataFrame({"row_id": row_id, "birds": birds})
        prediction_dfs.append(prediction_df)

    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df
