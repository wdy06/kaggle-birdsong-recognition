import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
import model_utils
import predict_utils
import utils

parser = argparse.ArgumentParser(description="global wheat detection")
parser.add_argument("--debug", help="run debug mode", action="store_true")
parser.add_argument("--cv", help="predict cross valid model", action="store_true")
parser.add_argument("--denoise", help="apply nosei reduction", action="store_true")
parser.add_argument("--th", "-t", help="threshold", type=float, default=None)
parser.add_argument("--model_dir", "-m", help="model path", type=str, required=True)
args = parser.parse_args()

# SAMPLE_RATE = 32000
# IMAGE_SIZE = 224

# submission.csv will be overwitten if everything goes well
submission = pd.read_csv(utils.DATA_DIR / "sample_submission.csv")
submission["birds"] = "a"
submission.to_csv("submission.csv", index=False)

model_dir = Path(args.model_dir)
config_path = model_dir / ".hydra" / "config.yaml"
config = utils.load_yaml(config_path)

model_config_list = []
if args.cv:
    print("predict cv model")
    FOLD = 5
    if args.debug:
        FOLD = 1
    for i in range(FOLD):
        model_config = {
            "path": model_dir / f"best_model_fold{i}.pth",
            "model_name": config["model"]["name"],
            "n_class": len(utils.BIRD_CODE),
            "in_chans": config["model"]["in_chans"],
        }
        model_config_list.append(model_config)
else:
    print("predict all model")
    model_config = {
        "path": model_dir / f"all_model.pth",
        "model_name": config["model"]["name"],
        "n_class": len(utils.BIRD_CODE),
        "in_chans": config["model"]["in_chans"],
    }
    model_config_list.append(model_config)

if args.th:
    print(f"override threshold with {args.th}")
    threshold = args.th
else:
    threshold = utils.load_json(model_dir / "threshold.json")
# test_df = pd.read_csv(utils.DATA_DIR / "test.csv")
test_audio_dir = utils.DATA_DIR / "test_audio"
if test_audio_dir.exists():
    test_df = pd.read_csv(utils.DATA_DIR / "test.csv")
else:
    check_dir = Path("/kaggle/input/birdcall-check/")
    test_audio_dir = check_dir / "test_audio"
    test_df = pd.read_csv(check_dir / "test.csv")

composer = utils.build_composer(
    sample_rate=config["sample_rate"],
    img_size=config["image_size"],
    in_chans=config["model"]["in_chans"],
)
submission = predict_utils.prediction(
    test_df=test_df,
    test_audio=test_audio_dir,
    ds_class=datasets.SpectrogramDataset,
    model_list=model_config_list,
    composer=composer,
    sample_rate=config["sample_rate"],
    threshold=threshold,
    denoise=args.denoise,
)

submission.to_csv("submission.csv", index=False)
