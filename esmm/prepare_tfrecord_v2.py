#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-20 0:25
"""
Efficient tfrecords writer interface
"""


import os
import sys
import os.path as ops
import tensorflow as tf
import argparse
from multiprocessing import Manager
from multiprocessing import Process
import time
import tqdm
import pandas as pd
import json

_SAMPLE_INFO_QUEUE = Manager().Queue()
_SENTINEL = ("", [])


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset_dir', type=str, help='The origin csv files dataset_dir', default=None)
    parser.add_argument('-s', '--save_dir', type=str, help='The generated tfrecords save dir', default=None)
    parser.add_argument('-f', '--dataset_flag', type=str, help='dataset flag', default='train')
    return parser.parse_args()


def _is_valid_file_type(file):
    (filename, extension) = os.path.splitext(file)
    if extension == '.csv':
        return True
    else:
        return False


def _init_data_queue(dataset_dir, writer_process_nums, dataset_flag='train'):
    print('Start filling {:s} dataset sample information queue...'.format(dataset_flag))
    t_start = time.time() #开始处理，计时
    input_files = os.listdir(dataset_dir)

    for i in tqdm.tqdm(input_files):
        try:
            _SAMPLE_INFO_QUEUE.put(os.path.join(dataset_dir,i))
        except IndexError:
            print('Lexicon doesn\'t contain lexicon index {:d}'.format(i))
            continue
    for i in range(writer_process_nums): # 添加结束标志
        _SAMPLE_INFO_QUEUE.put(_SENTINEL)
    print('Complete filling dataset sample information queue[current size: {:d}], cost time: {:.5f}s'.format(
        _SAMPLE_INFO_QUEUE.qsize(), time.time() - t_start))


def _write_tfrecords(tfrecords_writer):
    while True:
        sample_info = _SAMPLE_INFO_QUEUE.get()
        if sample_info == _SENTINEL:
            print('Process {:d} finished writing work'.format(os.getpid()))
            tfrecords_writer.close()
            break

        if not _is_valid_file_type(sample_info):
            print('Image file: {} is not a valid file type'.format(sample_info))
            continue

        #print ("sample info: ", sample_info)
        df = pd.read_csv(sample_info, sep='\t')
        lg = len(df.keys())
        #print ("lg ", lg)

        columns_nm_list = list(df.columns)

        for i in range(len(df)):
            # 创建字典
            features = {}
            # 写入标量，类型Int64，由于是标量，所以"value=[scalars[i]]" 变成list
            features['uid'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[str(df['uid'][i]).encode(encoding='unicode-escape')]))
            features['room_id'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[str(df['room_id'][i]).encode(encoding='unicode-escape')]))
            features['anchor_level'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['anchor_level'][i]]))
            features['anchor_score'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['anchor_score'][i]]))
            features['is_top200_anchor'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[df['is_top200_anchor'][i]]))
            features['room_click_num_1_day'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[df['room_click_num_1_day'][i]]))
            features['room_click_num_7_day'] = tf.train.Feature(
                float_list=tf.train.FloatList(value=[df['room_click_num_7_day'][i]]))
            # labels
            features['cvr_label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['cvr_label'][i]]))
            features['ctr_label'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['ctr_label'][i]]))

            # 将存有所有feature的字典送入tf.train.Features中
            #print ("features: ", features)
            tf_features = tf.train.Features(feature=features)
            #print ("tf_features: ", tf_features)
            # 再将其变成一个样本example
            tf_example = tf.train.Example(features=tf_features)
            # 序列化该样本
            tf_serialized = tf_example.SerializeToString()
            # write tfrecord
            tfrecords_writer.write(tf_serialized)


class CrnnDataProducer(object):

    def __init__(self, dataset_dir, char_dict_path=None, anno_file_path=None,
                 writer_process_nums=4, dataset_flag='train'):
        if not ops.exists(dataset_dir):
            raise ValueError('Dataset dir {:s} not exist'.format(dataset_dir))

        # Check image source data
        self._dataset_dir = dataset_dir
        self._annotation_file_path = anno_file_path
        self._char_dict_path = char_dict_path
        self._writer_process_nums = writer_process_nums
        self._dataset_flag = dataset_flag

    def generate_tfrecords(self, save_dir):
        # make save dirs
        os.makedirs(save_dir, exist_ok=True)
        # generate training example tfrecords
        print('Generating training sample tfrecords...')
        t_start = time.time()
        print('Start write tensorflow records for {:s}...'.format(self._dataset_flag))

        process_pool = []
        tfwriters = []
        for i in range(self._writer_process_nums):
            tfrecords_save_name = '{:s}_{:d}.tfrecords'.format(self._dataset_flag, i + 1)
            tfrecords_save_path = ops.join(save_dir, tfrecords_save_name)

            tfrecords_io_writer = tf.python_io.TFRecordWriter(path=tfrecords_save_path)
            process = Process(target=_write_tfrecords, name='Subprocess_{:d}'.format(i + 1),
                args=(tfrecords_io_writer,))
            process_pool.append(process)
            tfwriters.append(tfrecords_io_writer)
            process.start()

        for process in process_pool:
            process.join()
        print('Generate {:s} sample tfrecords complete, cost time: {:.5f}'\
              .format(self._dataset_flag, time.time() - t_start))
        return


def write_tfrecords(dataset_dir, save_dir, dataset_flag, writer_process_nums):
    assert ops.exists(dataset_dir), '{:s} not exist'.format(dataset_dir)
    os.makedirs(save_dir, exist_ok=True)
    # test data producer
    _init_data_queue(dataset_dir=dataset_dir, writer_process_nums=writer_process_nums, dataset_flag=dataset_flag)
    producer = CrnnDataProducer(dataset_dir=dataset_dir,  writer_process_nums=writer_process_nums, dataset_flag=dataset_flag)
    producer.generate_tfrecords(save_dir=save_dir)

if __name__ == '__main__':
    args = init_args()

    write_tfrecords(dataset_dir=args.dataset_dir,
                    save_dir=args.save_dir,
                    dataset_flag=args.dataset_flag,
                    writer_process_nums = 20)


