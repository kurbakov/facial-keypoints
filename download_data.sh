#!/bin/bash

rm -fr ./data
mkdir ./data
wget -P ./data/ https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip
unzip -n ./data/train-test-data.zip -d ./data