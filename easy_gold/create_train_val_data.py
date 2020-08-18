from collections import defaultdict
import librosa
import soundfile as sf
from pathlib import Path
import pandas as pd

import utils

train = pd.read_csv(utils.DATA_DIR / 'train.csv')
classes = train.ebird_code.unique().tolist()

recs = defaultdict(list)
for directory in Path(utils.DATA_DIR / 'train_resampled').iterdir():
    ebird_code = directory.name
    for file in directory.iterdir():
        recs[ebird_code].append((file, sf.info(file).duration))

train, val = {}, {}

for ebird in recs.keys():
    rs = recs[ebird]
    val_count = max(int(len(rs) * 0.1), 1)

    val[ebird] = rs[:val_count]
    train[ebird] = rs[val_count:]

pd.to_pickle(classes, utils.DATA_DIR / 'classes.pkl')
pd.to_pickle(train, utils.DATA_DIR / 'train_set.pkl')
pd.to_pickle(val, utils.DATA_DIR / 'val_set.pkl')
pd.to_pickle(recs, utils.DATA_DIR / 'recs.pkl')
