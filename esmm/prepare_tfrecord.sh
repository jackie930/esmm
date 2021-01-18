#!/usr/bin/env bash

#download train and test data from s3
mkdir -p ./dy_train
mkdir -p ./dy_test
mkidr -p ./data_tfrecord

aws s3 cp s3://douyu-datalab/train_1221.csv/ ./dy_train --recursive
aws s3 cp s3://douyu-datalab/test_1221.csv/ ./dy_test --recursive

#process train
python prepare_tfrecord.py --output_path='./data_tfrecord./train.tfrecord' --input_folder='./dy_train'
#process test
python prepare_tfrecord.py --output_path='./data_tfrecord/test.tfrecord' --input_folder='./dy_test'
