import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from numpy import lib
from torch.utils.data import Dataset

import librosa
import soundfile as sf


def audio_to_spec(audio, sample_rate):
    f, t, spec = scipy.signal.spectrogram(audio, fs=sample_rate)  # , nperseg=360)
    spec = np.log10(spec.clip(1e-10))
    return spec[10:100]
    spec = librosa.power_to_db(
        #         librosa.feature.melspectrogram(audio, sr=SAMPLE_RATE, fmin=20, fmax=16000, n_mels=224, hop_length=360)
        librosa.feature.melspectrogram(
            audio, sr=sample_rate, fmin=20, fmax=16000, n_mels=128
        )
        #         librosa.feature.melspectrogram(audio, sr=SAMPLE_RATE, hop_length=1255, fmin=20, fmax=16000)
    )
    return spec


class SpectrogramDataset(Dataset):
    def __init__(
        self, recs, classes, sample_rate, len_mult=20, spec_min=None, spec_max=None
    ):
        self.recs = recs
        self.vocab = classes
        self.sample_rate = sample_rate
        self.do_norm = spec_min and spec_max
        self.spec_min = spec_min
        self.spec_max = spec_max
        self.len_mult = len_mult

    def __getitem__(self, idx):
        warnings.filterwarnings("ignore")
        cls_idx = idx % len(self.vocab)
        recs = self.recs[self.vocab[cls_idx]]
        path, duration = recs[np.random.randint(0, len(recs))]
        if duration > 5:
            # x, _ = sf.read(
            #     path,
            #     start=int(np.random.rand() * (duration - 5) * self.sample_rate),
            #     frames=5 * self.sample_rate,
            #     samplerate=self.sample_rate,
            # )
            x, _ = librosa.load(
                path,
                mono=True,
                offset=int(np.random.rand() * (duration - 5)),
                duration=5,
                sr=self.sample_rate,
            )
        else:
            # x, _ = sf.read(path, samplerate=self.sample_rate)
            x, _ = librosa.load(path, sr=self.sample_rate)
            x = np.tile(x, 15)  # the shortest rec in the train set is 0.39 sec
            start_frame = int(np.random.rand() * (x.shape[0] - 5 * self.sample_rate))
            x = x[start_frame : start_frame + 5 * self.sample_rate]
        # print(x.shape)
        if x.shape[0] != 5 * self.sample_rate:
            raise Exception(f"Incorrect length: {x.shape[0]}, {path}, {duration}")
        x = audio_to_spec(x, self.sample_rate)
        if self.do_norm:
            x = self.normalize(x)
        img = np.repeat(x[None, :, :], 3, 0)
        return img.astype(np.float32), self.one_hot_encode(cls_idx)

    def normalize(self, x):
        return ((x - x.min()) / (x.max() - x.min() + 1e-8) - 0.11754986) / 0.16654329
        # x = (x - self.spec_min) / (self.spec_max - self.spec_min)
        # return (x - 0.36829123) / 0.08813263

    def show(self, idx):
        x = self[idx][0]
        x = (x * 0.36829123) + 0.08813263
        return plt.imshow(x.transpose(1, 2, 0)[:, :, 0])

    def one_hot_encode(self, y):
        one_hot = np.zeros((len(self.vocab)))
        one_hot[y] = 1
        return one_hot

    def __len__(self):
        return self.len_mult * len(self.vocab)


class TestSpectrogramDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, clip: np.ndarray, sample_rate, spec_min, spec_max
    ):
        self.df = df
        self.clip = clip
        self.sample_rate = sample_rate
        self.do_norm = spec_min and spec_max
        self.spec_min = spec_min
        self.spec_max = spec_max

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        #         SR = 32000
        sample = self.df.loc[idx, :]
        site = sample.site
        row_id = sample.row_id

        if site == "site_3":
            y = self.clip.astype(np.float32)
            len_y = len(y)
            start = 0
            end = self.sample_rate * 5
            images = []
            while len_y > start:
                y_batch = y[start:end].astype(np.float32)
                if len(y_batch) != (self.sample_rate * 5):
                    break
                start = end
                end = end + self.sample_rate * 5

                y = audio_to_spec(y, self.sample_rate)
                if self.do_norm:
                    y = self.normalize(y)
                image = np.repeat(y[None, :, :], 3, 0)
                images.append(image)
            images = np.asarray(images)
            return images, row_id, site
        else:
            end_seconds = int(sample.seconds)
            start_seconds = int(end_seconds - 5)

            start_index = self.sample_rate * start_seconds
            end_index = self.sample_rate * end_seconds

            y = self.clip[start_index:end_index].astype(np.float32)
            y = audio_to_spec(y, self.sample_rate)
            if self.do_norm:
                y = self.normalize(y)
            image = np.repeat(y[None, :, :], 3, 0)

            return image, row_id, site

    def normalize(self, x):
        return ((x - x.min()) / (x.max() - x.min() + 1e-8) - 0.11754986) / 0.16654329
