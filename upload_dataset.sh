#!/bin/bash

echo $1
EXP_NAME=$1
kaggle datasets init -p outputs/$EXP_NAME
sed -i -e s/INSERT_TITLE_HERE/$EXP_NAME/ outputs/$EXP_NAME/dataset-metadata.json
sed -i -e s/INSERT_SLUG_HERE/$EXP_NAME/ outputs/$EXP_NAME/dataset-metadata.json
kaggle datasets create -p outputs/$EXP_NAME -r zip

