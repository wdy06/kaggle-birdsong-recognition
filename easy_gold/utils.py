import json
import os
import pickle
import random
import warnings
from datetime import datetime
from pathlib import Path

import cv2
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

BIRD_CODE = {
    "aldfly": 0,
    "ameavo": 1,
    "amebit": 2,
    "amecro": 3,
    "amegfi": 4,
    "amekes": 5,
    "amepip": 6,
    "amered": 7,
    "amerob": 8,
    "amewig": 9,
    "amewoo": 10,
    "amtspa": 11,
    "annhum": 12,
    "astfly": 13,
    "baisan": 14,
    "baleag": 15,
    "balori": 16,
    "banswa": 17,
    "barswa": 18,
    "bawwar": 19,
    "belkin1": 20,
    "belspa2": 21,
    "bewwre": 22,
    "bkbcuc": 23,
    "bkbmag1": 24,
    "bkbwar": 25,
    "bkcchi": 26,
    "bkchum": 27,
    "bkhgro": 28,
    "bkpwar": 29,
    "bktspa": 30,
    "blkpho": 31,
    "blugrb1": 32,
    "blujay": 33,
    "bnhcow": 34,
    "boboli": 35,
    "bongul": 36,
    "brdowl": 37,
    "brebla": 38,
    "brespa": 39,
    "brncre": 40,
    "brnthr": 41,
    "brthum": 42,
    "brwhaw": 43,
    "btbwar": 44,
    "btnwar": 45,
    "btywar": 46,
    "buffle": 47,
    "buggna": 48,
    "buhvir": 49,
    "bulori": 50,
    "bushti": 51,
    "buwtea": 52,
    "buwwar": 53,
    "cacwre": 54,
    "calgul": 55,
    "calqua": 56,
    "camwar": 57,
    "cangoo": 58,
    "canwar": 59,
    "canwre": 60,
    "carwre": 61,
    "casfin": 62,
    "caster1": 63,
    "casvir": 64,
    "cedwax": 65,
    "chispa": 66,
    "chiswi": 67,
    "chswar": 68,
    "chukar": 69,
    "clanut": 70,
    "cliswa": 71,
    "comgol": 72,
    "comgra": 73,
    "comloo": 74,
    "commer": 75,
    "comnig": 76,
    "comrav": 77,
    "comred": 78,
    "comter": 79,
    "comyel": 80,
    "coohaw": 81,
    "coshum": 82,
    "cowscj1": 83,
    "daejun": 84,
    "doccor": 85,
    "dowwoo": 86,
    "dusfly": 87,
    "eargre": 88,
    "easblu": 89,
    "easkin": 90,
    "easmea": 91,
    "easpho": 92,
    "eastow": 93,
    "eawpew": 94,
    "eucdov": 95,
    "eursta": 96,
    "evegro": 97,
    "fiespa": 98,
    "fiscro": 99,
    "foxspa": 100,
    "gadwal": 101,
    "gcrfin": 102,
    "gnttow": 103,
    "gnwtea": 104,
    "gockin": 105,
    "gocspa": 106,
    "goleag": 107,
    "grbher3": 108,
    "grcfly": 109,
    "greegr": 110,
    "greroa": 111,
    "greyel": 112,
    "grhowl": 113,
    "grnher": 114,
    "grtgra": 115,
    "grycat": 116,
    "gryfly": 117,
    "haiwoo": 118,
    "hamfly": 119,
    "hergul": 120,
    "herthr": 121,
    "hoomer": 122,
    "hoowar": 123,
    "horgre": 124,
    "horlar": 125,
    "houfin": 126,
    "houspa": 127,
    "houwre": 128,
    "indbun": 129,
    "juntit1": 130,
    "killde": 131,
    "labwoo": 132,
    "larspa": 133,
    "lazbun": 134,
    "leabit": 135,
    "leafly": 136,
    "leasan": 137,
    "lecthr": 138,
    "lesgol": 139,
    "lesnig": 140,
    "lesyel": 141,
    "lewwoo": 142,
    "linspa": 143,
    "lobcur": 144,
    "lobdow": 145,
    "logshr": 146,
    "lotduc": 147,
    "louwat": 148,
    "macwar": 149,
    "magwar": 150,
    "mallar3": 151,
    "marwre": 152,
    "merlin": 153,
    "moublu": 154,
    "mouchi": 155,
    "moudov": 156,
    "norcar": 157,
    "norfli": 158,
    "norhar2": 159,
    "normoc": 160,
    "norpar": 161,
    "norpin": 162,
    "norsho": 163,
    "norwat": 164,
    "nrwswa": 165,
    "nutwoo": 166,
    "olsfly": 167,
    "orcwar": 168,
    "osprey": 169,
    "ovenbi1": 170,
    "palwar": 171,
    "pasfly": 172,
    "pecsan": 173,
    "perfal": 174,
    "phaino": 175,
    "pibgre": 176,
    "pilwoo": 177,
    "pingro": 178,
    "pinjay": 179,
    "pinsis": 180,
    "pinwar": 181,
    "plsvir": 182,
    "prawar": 183,
    "purfin": 184,
    "pygnut": 185,
    "rebmer": 186,
    "rebnut": 187,
    "rebsap": 188,
    "rebwoo": 189,
    "redcro": 190,
    "redhea": 191,
    "reevir1": 192,
    "renpha": 193,
    "reshaw": 194,
    "rethaw": 195,
    "rewbla": 196,
    "ribgul": 197,
    "rinduc": 198,
    "robgro": 199,
    "rocpig": 200,
    "rocwre": 201,
    "rthhum": 202,
    "ruckin": 203,
    "rudduc": 204,
    "rufgro": 205,
    "rufhum": 206,
    "rusbla": 207,
    "sagspa1": 208,
    "sagthr": 209,
    "savspa": 210,
    "saypho": 211,
    "scatan": 212,
    "scoori": 213,
    "semplo": 214,
    "semsan": 215,
    "sheowl": 216,
    "shshaw": 217,
    "snobun": 218,
    "snogoo": 219,
    "solsan": 220,
    "sonspa": 221,
    "sora": 222,
    "sposan": 223,
    "spotow": 224,
    "stejay": 225,
    "swahaw": 226,
    "swaspa": 227,
    "swathr": 228,
    "treswa": 229,
    "truswa": 230,
    "tuftit": 231,
    "tunswa": 232,
    "veery": 233,
    "vesspa": 234,
    "vigswa": 235,
    "warvir": 236,
    "wesblu": 237,
    "wesgre": 238,
    "weskin": 239,
    "wesmea": 240,
    "wessan": 241,
    "westan": 242,
    "wewpew": 243,
    "whbnut": 244,
    "whcspa": 245,
    "whfibi": 246,
    "whtspa": 247,
    "whtswi": 248,
    "wilfly": 249,
    "wilsni1": 250,
    "wiltur": 251,
    "winwre3": 252,
    "wlswar": 253,
    "wooduc": 254,
    "wooscj2": 255,
    "woothr": 256,
    "y00475": 257,
    "yebfly": 258,
    "yebsap": 259,
    "yehbla": 260,
    "yelwar": 261,
    "yerwar": 262,
    "yetvir": 263,
}

INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}


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

            with torch.no_grad():
                prediction = model(image)
                #                 proba = prediction["multilabel_proba"].detach().cpu().numpy().reshape(-1)
                proba = prediction.detach().cpu().numpy().reshape(-1)

            events = proba >= threshold
            labels = np.argwhere(events).reshape(-1).tolist()

        else:
            # to avoid prediction on large batch
            y = clip.astype(np.float32)
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
            labels_str_list = list(map(lambda x: INV_BIRD_CODE[x], labels))
            label_string = " ".join(labels_str_list)
            prediction_dict[row_id] = label_string
    return prediction_dict


def prediction(
    test_df: pd.DataFrame,
    test_audio: Path,
    ds_class,
    model,
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
            model=model,
            composer=composer,
            threshold=threshold,
        )
        row_id = list(prediction_dict.keys())
        birds = list(prediction_dict.values())
        prediction_df = pd.DataFrame({"row_id": row_id, "birds": birds})
        prediction_dfs.append(prediction_df)

    prediction_df = pd.concat(prediction_dfs, axis=0, sort=False).reset_index(drop=True)
    return prediction_df


def mono_to_color(
    X: np.ndarray, mean=None, std=None, norm_max=None, norm_min=None, eps=1e-6
):
    # Stack X as [X,X,X]
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    X = X - mean
    std = std or X.std()
    Xstd = X / (std + eps)
    _min, _max = Xstd.min(), Xstd.max()
    norm_max = norm_max or _max
    norm_min = norm_min or _min
    if (_max - _min) > eps:
        # Normalize to [0, 255]
        V = Xstd
        V[V < norm_min] = norm_min
        V[V > norm_max] = norm_max
        V = 255 * (V - norm_min) / (norm_max - norm_min)
        V = V.astype(np.uint8)
    else:
        # Just zero
        V = np.zeros_like(Xstd, dtype=np.uint8)
    return V


def build_composer(
    sample_rate,
    img_size,
    melspectrogram_parameters={},
    waveform_transforms=None,
    spectrogram_transforms=None,
):
    def composer(x):
        if waveform_transforms:
            x = waveform_transforms(x)
        melspec = librosa.feature.melspectrogram(
            x, sr=sample_rate, **melspectrogram_parameters
        )
        melspec = librosa.power_to_db(melspec).astype(np.float32)
        if spectrogram_transforms:
            melspec = spectrogram_transforms(melspec)

        image = mono_to_color(melspec)
        height, width, _ = image.shape
        image = cv2.resize(image, (int(width * img_size / height), img_size))
        image = np.moveaxis(image, 2, 0)
        image = (image / 255.0).astype(np.float32)
        return image

    return composer