import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import datasets
import model_utils
import utils

parser = argparse.ArgumentParser(description="global wheat detection")
parser.add_argument("--debug", help="run debug mode", action="store_true")
parser.add_argument("--model_dir", "-m", help="model path", type=str, required=True)

# SAMPLE_RATE = 32000
# IMAGE_SIZE = 224

# submission.csv will be overwitten if everything goes well
submission = pd.read_csv(utils.DATA_DIR / "sample_submission.csv")
submission["birds"] = "a"
submission.to_csv("submission.csv", index=False)

args = parser.parse_args()
model_dir = Path(args.model_dir)
config_path = model_dir / ".hydra" / "config.yaml"
config = utils.load_yaml(config_path)

model_config_list = []
model_config = {
    "path": model_dir / "best_model.pth",
    "model_name": config["model"]["name"],
    "n_class": len(utils.BIRD_CODE),
}
model_config_list.append(model_config)
# model_path = model_dir / "best_model.pth"
# model = model_utils.load_pytorch_model(
#     model_name=config["model"]["name"], path=model_path, n_class=len(utils.BIRD_CODE)
# )

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
    sample_rate=config["sample_rate"], img_size=config["image_size"]
)
submission = utils.prediction(
    test_df=test_df,
    test_audio=test_audio_dir,
    ds_class=datasets.SpectrogramDataset,
    model_list=model_config_list,
    composer=composer,
    sample_rate=config["sample_rate"],
    threshold=threshold,
)

submission.to_csv("submission.csv", index=False)
