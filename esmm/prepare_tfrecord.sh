#!/usr/bin/env bash

#download train and test data from s3
mkdir -p ./dy_train
mkdir -p ./dy_test

aws s3 cp s3://douyu-datalab/train_1221.csv/ ./dy_train --recursive
aws s3 cp s3://douyu-datalab/test_1221.csv/ ./dy_test --recursive

#process train v1
#python prepare_tfrecord.py --output_path='./data_tfrecord./train.tfrecord' --input_folder='./dy_train'
#process test v1
#python prepare_tfrecord.py --output_path='./data_tfrecord/test.tfrecord' --input_folder='./dy_test'

#process train
echo -e "training data prepare"

#process train v2
python prepare_tfrecord_v2.py --dataset_dir='./dy_full_data/dy_train' --save_dir='./dy_tfrecord_train_full' --dataset_flag='train'

#process train
echo -e "test data prepare"

#process test
python prepare_tfrecord_v2.py --dataset_dir='./dy_full_data/dy_test' --save_dir='./dy_tfrecord_test_full' --dataset_flag='test'
