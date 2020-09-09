import argparse
import os
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.filters.rank.generic import threshold

import librosa
import noisereduce as nr
import soundfile as sf
import utils


def convert_wav2npy(arg_tuple):
    path = arg_tuple[0]
    save_dir = arg_tuple[1]
    sample_rate = arg_tuple[2]
    denoise = arg_tuple[3]
    x = librosa.load(path, sr=sample_rate, mono=True)[0]
    if denoise:
        try:
            mask, env = utils.envelope(x, sample_rate, threshold=0.25)
            x = nr.reduce_noise(
                audio_clip=x, noise_clip=x[np.logical_not(mask)], verbose=False
            )
        except ValueError:
            print("=" * 10)
            print(path)
            print(x)
            # raise
    ebird_code = path.parent.name
    write_path = Path(save_dir) / f"{ebird_code}/{path.stem}.npy"
    print(f"write to {write_path}")
    # sf.write(write_path, x, sample_rate)
    np.save(write_path, x)


parser = argparse.ArgumentParser(description="resample")
parser.add_argument("--dry-run", help="dry run", action="store_true")
parser.add_argument("--denoise", help="do noise reduction", action="store_true")
parser.add_argument("--source_dir", help="source dir path", type=str, required=True)
parser.add_argument("--target_dir", help="target dir path", type=str, required=True)
args = parser.parse_args()

warnings.filterwarnings("ignore", category=UserWarning)
if args.denoise:
    print("do noise reduction")

NUM_WORKERS = 96
SAMPLE_RATE = 32000

# create directory
for directory in Path(args.source_dir).iterdir():
    ebird_code = directory.name
    if args.dry_run:
        print(f"{Path(args.target_dir) / str(ebird_code)}")
    else:
        if not (Path(args.target_dir) / str(ebird_code)).exists():
            os.makedirs(Path(args.target_dir) / str(ebird_code))


# resample
for directory in Path(args.source_dir).iterdir():
    file_paths = list(directory.iterdir())
    if args.dry_run:
        print(file_paths)
    else:
        file_paths = [
            (path, args.target_dir, SAMPLE_RATE, args.denoise) for path in file_paths
        ]
        with Pool(NUM_WORKERS // 2) as p:
            p.map(convert_wav2npy, file_paths)
