#!/bin/bash

echo $1
EXP_NAME=$1
kaggle datasets init -p results/$EXP_NAME
sed -i -e s/INSERT_TITLE_HERE/$EXP_NAME/ results/$EXP_NAME/dataset-metadata.json
sed -i -e s/INSERT_SLUG_HERE/$EXP_NAME/ results/$EXP_NAME/dataset-metadata.json
kaggle datasets create -p results/$EXP_NAME
