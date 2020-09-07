import base64
import gzip
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system("export PYTHONPATH=${PYTHONPATH}:/kaggle/working && " + command)


model_dir = ""
run("pip install ../input/tim-package-new/timm-0.1.30-py3-none-any.whl")
run("pip install ../input/resnest/resnest-0.0.5-py3-none-any.whl")
run("python setup.py develop --install-dir /kaggle/working")
run(f"python easy_gold/predict.py -m {model_dir} --th 0.6")
