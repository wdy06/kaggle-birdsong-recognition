import glob
import os
import random

import cv2
import IPython.display as ipd

# visuals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# albumentations core
import albumentations

# sound
import librosa
from albumentations.core.transforms_interface import BasicTransform, DualTransform


# ref: https://www.kaggle.com/tanulsingh077/audio-albumentations-transform-your-audio
class AudioTransform(BasicTransform):
    """ Transform for audio task. This is the main class where we override the targets and update params function for our need"""

    @property
    def targets(self):
        return {"data": self.apply}

    def update_params(self, params, **kwargs):
        if hasattr(self, "interpolation"):
            params["interpolation"] = self.interpolation
        if hasattr(self, "fill_value"):
            params["fill_value"] = self.fill_value
        return params


class TimeShifting(AudioTransform):
    """ Do time shifting of audio """

    def __init__(self, always_apply=False, p=0.5):
        super(TimeShifting, self).__init__(always_apply, p)

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        start_ = int(np.random.uniform(-80000, 80000))
        if start_ >= 0:
            audio_time_shift = np.r_[
                data[start_:], np.random.uniform(-0.001, 0.001, start_)
            ]
        else:
            audio_time_shift = np.r_[
                np.random.uniform(-0.001, 0.001, -start_), data[:start_]
            ]

        return audio_time_shift


class SpeedTuning(AudioTransform):
    """ Do speed Tuning of audio """

    def __init__(self, always_apply=False, p=0.5, speed_rate=None):
        """
        Give Rate between (0.5,1.5) for best results
        """
        super(SpeedTuning, self).__init__(always_apply, p)

        if speed_rate:
            self.speed_rate = speed_rate
        else:
            self.speed_rate = np.random.uniform(0.6, 1.3)

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        audio_speed_tune = cv2.resize(
            data, (1, int(len(data) * self.speed_rate))
        ).squeeze()
        if len(audio_speed_tune) < len(data):
            pad_len = len(data) - len(audio_speed_tune)
            audio_speed_tune = np.r_[
                np.random.uniform(-0.001, 0.001, int(pad_len / 2)),
                audio_speed_tune,
                np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2))),
            ]
        else:
            cut_len = len(audio_speed_tune) - len(data)
            audio_speed_tune = audio_speed_tune[
                int(cut_len / 2) : int(cut_len / 2) + len(data)
            ]

        return audio_speed_tune


class StretchAudio(AudioTransform):
    """ Do stretching of audio file"""

    def __init__(self, always_apply=False, p=0.5, rate=None):
        super(StretchAudio, self).__init__(always_apply, p)

        if rate:
            self.rate = rate
        else:
            self.rate = np.random.uniform(0.5, 1.5)

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        input_length = len(data)

        data = librosa.effects.time_stretch(data, self.rate)

        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

        return data


class PitchShift(AudioTransform):
    """ Do time shifting of audio """

    def __init__(self, always_apply=False, p=0.5, n_steps=None):
        super(PitchShift, self).__init__(always_apply, p)
        """
        nsteps here is equal to number of semitones
        """

        self.n_steps = n_steps

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        # return librosa.effects.pitch_shift(data, sr=22050, n_steps=self.n_steps)
        return librosa.effects.pitch_shift(data, sr=32000, n_steps=self.n_steps)


class AddGaussianNoise(AudioTransform):
    """ Do time shifting of audio """

    def __init__(self, always_apply=False, p=0.5):
        super(AddGaussianNoise, self).__init__(always_apply, p)

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        noise = np.random.randn(len(data))
        data_wn = data + 0.005 * noise
        return data_wn


class AddCustomNoise(AudioTransform):
    """
    This Function allows you to add noise from any custom file you want just give path to the directory where the files
    are stored and you are good to go.
    """

    def __init__(self, file_dir, always_apply=False, p=0.5):
        super(AddCustomNoise, self).__init__(always_apply, p)
        """
        file_dir must be of form '.../input/.../something'
        """

        self.noise_files = glob.glob(file_dir + "/*")

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        nf = self.noise_files[int(np.random.uniform(0, len(self.noise_files)))]

        noise, _ = librosa.load(nf)

        if len(noise) > len(data):
            start_ = np.random.randint(len(noise) - len(data))
            noise = noise[start_ : start_ + len(data)]
        else:
            noise = np.pad(noise, (0, len(data) - len(noise)), "constant")

        data_wn = data + noise

        return data_wn


class PolarityInversion(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(PolarityInversion, self).__init__(always_apply, p)

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        return -data


class Gain(AudioTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.
    """

    def __init__(
        self, min_gain_in_db=-12, max_gain_in_db=12, always_apply=False, p=0.5
    ):
        super(Gain, self).__init__(always_apply, p)
        assert min_gain_in_db <= max_gain_in_db
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db

    def apply(self, data, **args):
        amplitude_ratio = 10 ** (
            random.uniform(self.min_gain_in_db, self.max_gain_in_db) / 20
        )
        return data * amplitude_ratio


class CutOut(AudioTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(CutOut, self).__init__(always_apply, p)

    def apply(self, data, **params):
        """
        data : ndarray of audio timeseries
        """
        start_ = np.random.randint(0, len(data))
        end_ = np.random.randint(start_, len(data))

        data[start_:end_] = 0

        return data


def get_train_transforms(config):
    compose_list = []
    for aug_conf in config:
        transform = globals()[aug_conf["name"]](**aug_conf["param"])
        compose_list.append(transform)
    return albumentations.Compose(compose_list)

    # return albumentations.Compose(
    #     [
    #         TimeShifting(
    #             p=0.9
    #         ),  # here not p=1.0 because your nets should get some difficulties
    #         SpeedTuning(p=0.8),
    #         AddGaussianNoise(p=0.8),
    #         PitchShift(p=0.5, n_steps=4),
    #         Gain(p=0.9),
    #         PolarityInversion(p=0.9),
    #         StretchAudio(p=0.1),
    #     ]
    # )
