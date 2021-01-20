 # -*- coding:utf-8 -*-
import tensorflow as tf
import pandas as pd
import os

import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_path", help = "display a square of a given number")
parser.add_argument("-i", "--input_folder", help = "increase output verbosity")

def main(output_path,input_path):
    writer = tf.python_io.TFRecordWriter(output_path)
    df = pd.read_csv(input_path, sep='\t')
    lg = len(df.keys())

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

        # # 写入向量，类型float，本身就是list，所以"value=vectors[i]"没有中括号
        # features['vector'] = tf.train.Feature(float_list=tf.train.FloatList(value=vectors[i]))
        #
        # # 写入矩阵，类型float，本身是矩阵，一种方法是将矩阵flatten成list
        # features['matrix'] = tf.train.Feature(float_list=tf.train.FloatList(value=matrices[i].reshape(-1)))
        # # 然而矩阵的形状信息(2,3)会丢失，需要存储形状信息，随后可转回原形状
        # features['matrix_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=matrices[i].shape))

        # # 写入张量，类型float，本身是三维张量，另一种方法是转变成字符类型存储，随后再转回原类型
        # features['tensor'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[tensors[i].tostring()]))
        # # 存储丢失的形状信息(806,806,3)
        # features['tensor_shape'] = tf.train.Feature(int64_list=tf.train.Int64List(value=tensors[i].shape))
    # 将存有所有feature的字典送入tf.train.Features中
    tf_features = tf.train.Features(feature=features)
    # 再将其变成一个样本example
    tf_example = tf.train.Example(features=tf_features)
    # 序列化该样本
    tf_serialized = tf_example.SerializeToString()

    # 写入一个序列化的样本
    writer.write(tf_serialized)
    # 由于上面有循环3次，所以到此我们已经写了3个样本
    # 关闭文件
    writer.close()

def multi_files(input_folder,output_path):
    input_files = os.listdir(input_folder)
    leg = len(input_files)
    j = 0
    for i in input_files:
        (filename, extension) = os.path.splitext(i)
        #print ("filename: {}, extention {}". format(filename,extension))
        if extension=='.csv':
            main(output_path, os.path.join(input_folder,i))
            print('finish process {} / {}'.format(j,leg))
        j = j+1
    print ("process finished!")

if __name__ == "__main__":
    args = parser.parse_args()
    multi_files(args.input_folder,args.output_path)

