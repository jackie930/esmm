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
import numpy as np
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

def _convert_ratio(x):
    if type(x)==float:
        return 0
    else:
        return np.mean([float(i.split(":")[1]) for i in x.split(',')])

def _generate(col,cut_off):
    try:
        a_list = col.replace('[', '').replace(']', '').replace('\\','').replace('"','').split(',')
        b_list = [int(i.split(":")[0]) for i in a_list]

        if len(b_list)< int(cut_off):
            return b_list.extend([0]*((cut_off)-len(b_list)))
        elif len(b_list)==cut_off:
            return b_list
        elif len(b_list)> cut_off:
            return b_list[:cut_off]
    except:
        return [0]*(cut_off)

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
        # filter out when user id can't match
        df = df[df.effect_watch_cate_rooms_cnt_7d.notnull()]
        df['user_cate'] = df.apply(lambda row: _generate(row['cate'], 3), axis=1)
        df['user_tag'] = df.apply(lambda row: _generate(row['tag'], 3), axis=1)
        df['user_child'] = df.apply(lambda row: _generate(row['child'], 3), axis=1)

        # user_tag_favor_1_day 用户7天喜好程度按score分5个等级,top20分4等级,其他分位第5级（1,2,3,4,5）
        # 用户7天对分区的ctr转化率，top20分区
        df['user_tag_favor_1_day'] = df.apply(lambda row: _generate(row['user_tag_favor_1_day'], 3), axis=1)
        df['user_tag_favor_7_day'] = df.apply(lambda row: _generate(row['user_tag_favor_7_day'], 3), axis=1)
        df['user_tag_favor_15_day'] = df.apply(lambda row: _generate(row['user_tag_favor_15_day'], 3), axis=1)
        df['tag_ratio_1_day_avg'] = df['tag_ratio_1_day'].apply(lambda x: _convert_ratio(x))
        df['tag_ratio_7_day_avg'] = df['tag_ratio_7_day'].apply(lambda x: _convert_ratio(x))
        df['tag_ratio_15_day_avg'] = df['tag_ratio_15_day'].apply(lambda x: _convert_ratio(x))

        lg = len(df.keys())
        #print ("lg ", lg)

        columns_nm_list = list(df.columns)

        for i in range(len(df)):
            try:
                # 创建字典
                features = {}
                # 写入标量，类型Int64，由于是标量，所以"value=[scalars[i]]" 变成list
                # 写入标量，类型Int64，由于是标量，所以"value=[scalars[i]]" 变成list
                features['uid'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str((df['uid'][i])).encode(encoding='unicode-escape')]))
                features['room_id'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str((df['room_id'][i])).encode(encoding='unicode-escape')]))
                # item feature
                features['anchor_level'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['anchor_level'][i]]))
                features['anchor_score'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['anchor_score'][i]]))
                features['is_top200_anchor'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['is_top200_anchor'][i]]))
                features['room_click_num_1_day'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_click_num_1_day'][i]]))
                features['room_click_num_7_day'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_click_num_7_day'][i]]))
                features['room_click_num_15_day'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_click_num_15_day'][i]]))
                features['room_ctr_1_day'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_ctr_1_day'][i]]))
                features['room_ctr_7_day'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_ctr_7_day'][i]]))
                features['room_ctr_15_day'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_ctr_15_day'][i]]))
                # 房间在周一热度/ 房间在周五热度/ 房间在周末热度
                features['i_hot_mon'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['i_hot_mon'][i]]))
                features['i_hot_fri'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['i_hot_fri'][i]]))
                features['i_hot_weekend'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['i_hot_weekend'][i]]))
                features['i_hot_weekday'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['i_hot_weekday'][i]]))
                features['i_ctr_mon'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['i_ctr_mon'][i]]))
                features['i_ctr_fri'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['i_ctr_fri'][i]]))
                features['i_ctr_weekend'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['i_ctr_weekend'][i]]))
                features['i_ctr_weekday'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['i_ctr_weekday'][i]]))
                # 时段偏好(todo: do we need to update encoding style?)
                features['guess_cnt_1d'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['guess_cnt_1d'][i]]))
                features['guess_cnt_7d'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['guess_cnt_7d'][i]]))
                features['guess_cnt_15d'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['guess_cnt_15d'][i]]))
                features['raffle_cnt_1d'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['raffle_cnt_1d'][i]]))
                features['raffle_cnt_7d'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['raffle_cnt_7d'][i]]))
                features['raffle_cnt_15d'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['raffle_cnt_15d'][i]]))
                features['room_watch_newmbr_cnt_30d'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_watch_newmbr_cnt_30d'][i]]))
                features['room_effwatch_mbr_ratio_30d'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_effwatch_mbr_ratio_30d'][i]]))
                features['room_follow_mbr_ratio_30d'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_follow_mbr_ratio_30d'][i]]))
                features['room_click_repeat_uids_1d'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_click_repeat_uids_1d'][i]]))
                features['room_click_repeat_uids_7d'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_click_repeat_uids_7d'][i]]))
                features['room_click_repeat_uids_15d'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['room_click_repeat_uids_15d'][i]]))
                features['click_repeat_avg_nums_15d'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['click_repeat_avg_nums_15d'][i]]))
                # 房间重要特征
                features['item_age'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['item_age'][i]]))
                features['owner_level'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['owner_level'][i]]))
                features['item_sex'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str((df['item_sex'][i])).encode(encoding='unicode-escape')]))
                features['fans'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['fans'][i]]))
                # id(#共享层定义：cate-一级分区，tag二级分区，child三级分区)
                features['cate_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(df['cate_id'][i])]))
                features['child_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(df['child_id'][i])]))
                features['tag_id'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(df['tag_id'][i])]))
                # 星座
                features['constellation'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str((df['constellation'][i])).encode(encoding='unicode-escape')]))

                # user feature
                features['own_room'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(int(df['own_room'][i])).encode(encoding='unicode-escape')]))
                features['email_status'] = tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[str(int(df['email_status'][i])).encode(encoding='unicode-escape')]))
                features['phone_status'] = tf.train.Feature(bytes_list=tf.train.BytesList(
                    value=[str(int(df['phone_status'][i])).encode(encoding='unicode-escape')]))
                features['sex'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(int(df['sex'][i])).encode(encoding='unicode-escape')]))

                features['src'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(df['src'][i]).encode()]))
                features['city'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(df['city'][i]).encode()]))
                features['province'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(df['province'][i]).encode()]))
                features['brand'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(df['brand'][i]).encode()]))
                features['model'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(df['model'][i]).encode()]))

                features['active'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['active'][i]]))
                features['level'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['level'][i]]))
                features['days'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['days'][i]]))
                features['msg_cnt'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['msg_cnt'][i]]))

                features['effect_watch_days'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['effect_watch_days'][i]]))
                features['effect_watch_room_cnt'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['effect_watch_room_cnt'][i]]))
                features['watch_time'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['watch_time'][i]]))
                features['yuwan_tag'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(int(df['yuwan_tag'][i])).encode(encoding='unicode-escape')]))

                features['yuwan_cnt'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['yuwan_cnt'][i]]))

                features['follownum'] = tf.train.Feature(float_list=tf.train.FloatList(value=[df['follownum'][i]]))

                features['rich_tag'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(int(df['rich_tag'][i])).encode(encoding='unicode-escape')]))
                features['loser_tag'] = tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[str(int(df['loser_tag'][i])).encode(encoding='unicode-escape')]))

                # 用户分区属性
                features['user_cate'] = tf.train.Feature(int64_list=tf.train.Int64List(value=df['user_cate'][i]))
                features['user_tag'] = tf.train.Feature(int64_list=tf.train.Int64List(value=df['user_tag'][i]))
                features['user_child'] = tf.train.Feature(int64_list=tf.train.Int64List(value=df['user_child'][i]))

                # 偏好分区个数
                features['cate_length'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(df['cate_length'][i])]))
                features['tag_length'] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(df['tag_length'][i])]))
                features['child_length'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(df['child_length'][i])]))

                # 分区偏好
                features['user_tag_favor_1_day'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=df['user_tag_favor_1_day'][i]))
                features['user_tag_favor_7_day'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=df['user_tag_favor_7_day'][i]))
                features['user_tag_favor_15_day'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=df['user_tag_favor_15_day'][i]))
                # 偏好分区平均ctr
                features['tag_ratio_1_day_avg'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['tag_ratio_1_day_avg'][i]]))
                features['tag_ratio_7_day_avg'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['tag_ratio_7_day_avg'][i]]))
                features['tag_ratio_15_day_avg'] = tf.train.Feature(
                    float_list=tf.train.FloatList(value=[df['tag_ratio_15_day_avg'][i]]))

                # 活跃度
                features['u_active_mon'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(df['u_active_mon'][i])]))
                features['u_active_fri'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(df['u_active_fri'][i])]))
                features['u_active_weekday'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(df['u_active_weekday'][i])]))
                features['u_active_weekend'] = tf.train.Feature(
                    int64_list=tf.train.Int64List(value=[int(df['u_active_weekend'][i])]))

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
            except:
                pass

class ESMMDataProducer(object):

    def __init__(self, dataset_dir, writer_process_nums=4, dataset_flag='train'):
        if not ops.exists(dataset_dir):
            raise ValueError('Dataset dir {:s} not exist'.format(dataset_dir))

        # Check image source data
        self._dataset_dir = dataset_dir
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
    producer = ESMMDataProducer(dataset_dir=dataset_dir,  writer_process_nums=writer_process_nums, dataset_flag=dataset_flag)
    producer.generate_tfrecords(save_dir=save_dir)

if __name__ == '__main__':
    args = init_args()

    write_tfrecords(dataset_dir=args.dataset_dir,
                    save_dir=args.save_dir,
                    dataset_flag=args.dataset_flag,
                    writer_process_nums = 20)


