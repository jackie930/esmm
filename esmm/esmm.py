# -*- coding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import json
import tensorflow as tf
from tensorflow import feature_column as fc
# for python 2.x
#import sys
# reload(sys)
# sys.setdefaultencoding("utf-8")

flags = tf.app.flags
flags.DEFINE_string("model_dir", "./model_dir", "Base directory for the model.")
flags.DEFINE_string("output_model", "./model_output",
                    "Path to the training data.")
flags.DEFINE_string("train_data", "data/data.tfrecord",
                    "Directory for storing mnist data")
flags.DEFINE_string("eval_data", "data/data.tfrecord",
                    "Path to the evaluation data.")
flags.DEFINE_string("hidden_units", "512,256,128",
                    "Comma-separated list of number of units in each hidden layer of the NN")
flags.DEFINE_integer("train_steps", 10000,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("train_epochs", 10,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 1, "Training batch size")
flags.DEFINE_integer("shuffle_buffer_size", 10000,
                     "dataset shuffle buffer size")
flags.DEFINE_float("learning_rate", 0.00001, "Learning rate")
flags.DEFINE_float("dropout_rate", 0.25, "Drop out rate")
flags.DEFINE_integer("num_parallel_readers", 5,
                     "number of parallel readers for training data")
flags.DEFINE_integer("save_checkpoints_steps", 5000,
                     "Save checkpoints every this many steps")
flags.DEFINE_string("ps_hosts", "s-xiasha-10-2-176-43.hx:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "s-xiasha-10-2-176-42.hx:2223,s-xiasha-10-2-176-44.hx:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None, "job name: worker or ps")
flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_boolean("run_on_cluster", False,
                     "Whether the cluster info need to be passed in as input")

FLAGS = flags.FLAGS
my_feature_columns = []


def set_tfconfig_environ():
    if "TF_CLUSTER_DEF" in os.environ:
        cluster = json.loads(os.environ["TF_CLUSTER_DEF"])
        task_index = int(os.environ["TF_INDEX"])
        task_type = os.environ["TF_ROLE"]

        tf_config = dict()
        worker_num = len(cluster["worker"])
        if task_type == "ps":
            tf_config["task"] = {"index": task_index, "type": task_type}
            FLAGS.job_name = "ps"
            FLAGS.task_index = task_index
        else:
            if task_index == 0:
                tf_config["task"] = {"index": 0, "type": "chief"}
            else:
                tf_config["task"] = {"index": task_index - 1, "type": task_type}
            FLAGS.job_name = "worker"
            FLAGS.task_index = task_index

        if worker_num == 1:
            cluster["chief"] = cluster["worker"]
            del cluster["worker"]
        else:
            cluster["chief"] = [cluster["worker"][0]]
            del cluster["worker"][0]

        tf_config["cluster"] = cluster
        os.environ["TF_CONFIG"] = json.dumps(tf_config)
        print("TF_CONFIG", json.loads(os.environ["TF_CONFIG"]))

    if "INPUT_FILE_LIST" in os.environ:
        INPUT_PATH = json.loads(os.environ["INPUT_FILE_LIST"])
        if INPUT_PATH:
            print("input path:", INPUT_PATH)
            FLAGS.train_data = INPUT_PATH.get(FLAGS.train_data)
            FLAGS.eval_data = INPUT_PATH.get(FLAGS.eval_data)
        else:  # for ps
            print("load input path failed.")
            FLAGS.train_data = None
            FLAGS.eval_data = None


def parse_argument():
    if FLAGS.job_name is None or FLAGS.job_name == "":
        raise ValueError("Must specify an explicit `job_name`")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")

    print("job name = %s" % FLAGS.job_name)
    print("task index = %d" % FLAGS.task_index)
    os.environ["TF_ROLE"] = FLAGS.job_name
    os.environ["TF_INDEX"] = str(FLAGS.task_index)

    # Construct the cluster and start the server
    ps_spec = FLAGS.ps_hosts.split(",")
    worker_spec = FLAGS.worker_hosts.split(",")
    cluster = {"worker": worker_spec, "ps": ps_spec}
    os.environ["TF_CLUSTER_DEF"] = json.dumps(cluster)

# behaviorBids,behaviorC1ids,behaviorCids,behaviorSids,behaviorPids,
# bidWeights,c1idWeights,sidWeights,pidWeights
# productId,sellerId,brandId,cate1Id,cateId,
# matchScore,popScore,brandPrefer,cate2Prefer,catePrefer,sellerPrefer,matchType,position
# triggerNum,triggerRank,type,hour,phoneBrand,phoneResolution,phoneOs,tab


def create_feature_columns():
    # user feature
    uid = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("uid", 1000))
    own_room = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("own_room", 2))
    email_status = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("email_status", 2))
    phone_status = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("phone_status", 2))
    sex = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("sex", 3))
    city = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("city", 100))
    province = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("province", 100))
    brand = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("brand", 100))
    model = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("model", 1000))
    active = fc.numeric_column("active", default_value=5.0)
    level = fc.numeric_column("level", default_value=10)
    days = fc.numeric_column("days", default_value=20)
    msg_cnt = fc.numeric_column("msg_cnt", default_value=67)
    effective_watch_days = fc.numeric_column(
        "effect_watch_days", default_value=18)
    effective_watch_room_cnt = fc.numeric_column(
        "effect_watch_room_cnt", default_value=3.5)
    watch_time = fc.numeric_column("watch_time", default_value=130000)
    yuwan_tag = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("yuwan_tag", 100))
    yuwan_cnt = fc.numeric_column("yuwan_cnt", default_value=400)
    follownum = fc.numeric_column("follownum", default_value=54)
    rich_tag = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("rich_tag", 2))
    loser_tag = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("loser_tag", 2))
    user_cate = fc.categorical_column_with_hash_bucket(
        "user_cate", 10000, dtype=tf.int64)
    # user_cate = fc.indicator_column(
    #     fc.categorical_column_with_identity("user_cate", 101, default_value=100))
    user_tag = fc.categorical_column_with_hash_bucket(
        "user_tag", 10000, dtype=tf.int64)
    # user_tag = fc.indicator_column(
    #     fc.categorical_column_with_identity("user_tag", 101, default_value=100))
    user_child = fc.categorical_column_with_hash_bucket(
        "user_child", 10000, dtype=tf.int64)
    # user_child = fc.indicator_column(
    #     fc.categorical_column_with_identity("user_child", 101, default_value=100))
    cate_length = fc.indicator_column(
        fc.categorical_column_with_identity("cate_length", 11, default_value=10))
    tag_length = fc.indicator_column(
        fc.categorical_column_with_identity("tag_length", 36, default_value=35))
    child_length = fc.indicator_column(
        fc.categorical_column_with_identity("child_length", 101, default_value=100))
    user_tag_favor_1_day = fc.categorical_column_with_hash_bucket(
        "user_tag_favor_1_day", 10000, dtype=tf.int64)
    user_tag_favor_7_day = fc.categorical_column_with_hash_bucket(
        "user_tag_favor_7_day", 10000, dtype=tf.int64)
    user_tag_favor_15_day = fc.categorical_column_with_hash_bucket(
        "user_tag_favor_15_day", 10000, dtype=tf.int64)
    tag_ratio_1_day_avg = fc.numeric_column(
        "tag_ratio_1_day_avg", default_value=0.0)
    tag_ratio_7_day_avg = fc.numeric_column(
        "tag_ratio_7_day_avg", default_value=0.0)
    tag_ratio_15_day_avg = fc.numeric_column(
        "tag_ratio_15_day_avg", default_value=0.0)
    u_active_mon = fc.indicator_column(
        fc.categorical_column_with_identity("u_active_mon", 9, default_value=8))
    u_active_fri = fc.indicator_column(
        fc.categorical_column_with_identity("u_active_fri", 9, default_value=8))
    u_active_weekday = fc.indicator_column(
        fc.categorical_column_with_identity("u_active_weekday", 9, default_value=8))
    u_active_weekend = fc.indicator_column(
        fc.categorical_column_with_identity("u_active_weekend", 9, default_value=8))

    # item feature
    room_id = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("room_id", 1000))
    anchor_level = fc.numeric_column("anchor_level", default_value=68)
    anchor_score = fc.numeric_column("anchor_score", default_value=187326)
    is_top200_anchor = fc.numeric_column("is_top200_anchor", default_value=0.3)
    room_click_num_1_day = fc.numeric_column(
        "room_click_num_1_day", default_value=100000)
    room_click_num_7_day = fc.numeric_column(
        "room_click_num_7_day", default_value=450000)
    room_click_num_15_day = fc.numeric_column(
        "room_click_num_15_day", default_value=745762)
    room_ctr_1_day = fc.numeric_column(
        "room_ctr_1_day", default_value=0.2)
    room_ctr_7_day = fc.numeric_column(
        "room_ctr_7_day", default_value=0.2)
    room_ctr_15_day = fc.numeric_column(
        "room_ctr_15_day", default_value=0.2)
    i_hot_mon = fc.numeric_column(
        "i_hot_mon", default_value=850000)
    i_hot_fri = fc.numeric_column(
        "i_hot_fri", default_value=680000)
    i_hot_weekend = fc.numeric_column(
        "i_hot_weekend", default_value=2100000)
    i_hot_weekday = fc.numeric_column(
        "i_hot_weekday", default_value=1950000)
    i_ctr_mon = fc.numeric_column(
        "i_ctr_mon", default_value=0.2)
    i_ctr_weekend = fc.numeric_column(
        "i_ctr_weekend", default_value=0.2)
    i_ctr_weekday = fc.numeric_column(
        "i_ctr_weekday", default_value=0.2)
    guess_cnt_1d = fc.numeric_column(
        "guess_cnt_1d", default_value=4.0)
    guess_cnt_7d = fc.numeric_column(
        "guess_cnt_7d", default_value=15)
    guess_cnt_15d = fc.numeric_column(
        "guess_cnt_15d", default_value=27)
    raffle_cnt_1d = fc.numeric_column(
        "raffle_cnt_1d", default_value=0.9)
    raffle_cnt_7d = fc.numeric_column(
        "raffle_cnt_7d", default_value=6.0)
    raffle_cnt_15d = fc.numeric_column(
        "raffle_cnt_15d", default_value=12.5)
    room_watch_newmbr_cnt_30d = fc.numeric_column(
        "room_watch_newmbr_cnt_30d", default_value=238000)
    room_effwatch_mbr_ratio_30d = fc.numeric_column(
        "room_effwatch_mbr_ratio_30d", default_value=0.5)
    room_follow_mbr_ratio_30d = fc.numeric_column(
        "room_follow_mbr_ratio_30d", default_value=0.007)
    room_click_repeat_uids_1d = fc.numeric_column(
        "room_click_repeat_uids_1d", default_value=20000)
    room_click_repeat_uids_7d = fc.numeric_column(
        "room_click_repeat_uids_7d", default_value=75000)
    room_click_repeat_uids_15d = fc.numeric_column(
        "room_click_repeat_uids_15d", default_value=100000)
    click_repeat_avg_nums_15d = fc.numeric_column(
        "click_repeat_avg_nums_15d", default_value=5.0)
    item_age = fc.numeric_column(
        "item_age", default_value=25.7)
    owner_level = fc.numeric_column(
        "owner_level", default_value=68)
    item_sex = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("item_sex", 2))
    fans = fc.numeric_column(
        "fans", default_value=0.0)
    # cate_id = fc.indicator_column(
    #    fc.categorical_column_with_identity("cate_id", 21, default_value=20))
    cate_id = fc.categorical_column_with_hash_bucket(
        "cate_id", 10000, dtype=tf.int64)
    child_id = fc.categorical_column_with_hash_bucket(
        "child_id", 10000, dtype=tf.int64)
    # child_id = fc.indicator_column(
    #    fc.categorical_column_with_identity("child_id", 101, default_value=100))
#     tag_id = fc.categorical_column_with_identity("tag_id", 101, default_value=100)
    tag_id = fc.categorical_column_with_hash_bucket(
        "tag_id", 10000, dtype=tf.int64)
    constellation = fc.indicator_column(
        fc.categorical_column_with_hash_bucket("constellation", 12))

    # share feature
    user_tag_weighted = fc.weighted_categorical_column(
      user_tag, "user_tag_weight")
    user_tag_favor_1_day_weighted = fc.weighted_categorical_column(
      user_tag_favor_1_day, "user_tag_favor_1_day_weight")
    user_tag_favor_7_day_weighted = fc.weighted_categorical_column(
      user_tag_favor_7_day, "user_tag_favor_7_day_weight")
    user_tag_favor_15_day_weighted = fc.weighted_categorical_column(
      user_tag_favor_15_day, "user_tag_favor_15_day_weight")
    user_cate_weighted = fc.weighted_categorical_column(
      user_cate, "user_cate_weight")
    user_child_weighted = fc.weighted_categorical_column(
      user_child, "user_child_weight")

    # tag_id_embed = fc.shared_embedding_columns([user_tag_favor_1_day, user_tag_favor_7_day, user_tag_favor_15_day, user_tag,
    #                                             tag_id], 100, combiner='sum', shared_embedding_collection_name='tag_id_embed')
    # cate_id_embed = fc.shared_embedding_columns(
    #     [user_cate, cate_id], 100, combiner='sum', shared_embedding_collection_name='cate_id_embed')
    # child_id_embed = fc.shared_embedding_columns(
    #     [user_child, child_id], 100, combiner='sum', shared_embedding_collection_name='child_id_embed')
    tag_id_embed = fc.shared_embedding_columns([user_tag_favor_1_day, user_tag_favor_7_day, user_tag_favor_15_day, user_tag,
                                                tag_id], 10000, combiner='sum', shared_embedding_collection_name='tag_id_embed')
    cate_id_embed = fc.shared_embedding_columns(
        [user_cate, cate_id], 10000, combiner='sum', shared_embedding_collection_name='cate_id_embed')
    child_id_embed = fc.shared_embedding_columns(
        [user_child, child_id], 10000, combiner='sum', shared_embedding_collection_name='child_id_embed')

    # label
    click = fc.numeric_column("ctr_label", default_value=0.0)
    pay = fc.numeric_column("cvr_label", default_value=0.0)
    global my_feature_columns
    global my_label_feature_columns
    my_feature_columns = [uid, own_room, email_status, phone_status, sex, city, province, brand, model, active,
                          level, days, msg_cnt, effective_watch_days, effective_watch_room_cnt, watch_time, yuwan_tag, yuwan_cnt, follownum,
                          rich_tag, loser_tag, cate_length, tag_length, child_length, tag_ratio_1_day_avg,
                          tag_ratio_7_day_avg, tag_ratio_15_day_avg, u_active_mon, u_active_fri, u_active_weekday, u_active_weekend, room_id,
                          anchor_level, anchor_score, is_top200_anchor, room_click_num_1_day, room_click_num_7_day, room_click_num_15_day,
                          room_ctr_1_day, room_ctr_7_day, room_ctr_15_day, i_hot_mon, i_hot_fri, i_hot_weekend, i_hot_weekday, i_ctr_mon,
                          i_ctr_weekend, i_ctr_weekday, guess_cnt_1d, guess_cnt_7d, guess_cnt_15d, raffle_cnt_1d, raffle_cnt_7d, raffle_cnt_15d,
                          room_watch_newmbr_cnt_30d, room_effwatch_mbr_ratio_30d, room_follow_mbr_ratio_30d, room_click_repeat_uids_1d, room_click_repeat_uids_7d,
                          room_click_repeat_uids_15d, click_repeat_avg_nums_15d, item_age, owner_level, item_sex, fans, constellation]
    my_feature_columns += tag_id_embed
    my_feature_columns += cate_id_embed
    my_feature_columns += child_id_embed
    my_label_feature_columns = [uid, own_room, email_status, phone_status, sex, city, province, brand, model, active,
                                level, days, msg_cnt, effective_watch_days, effective_watch_room_cnt, watch_time, yuwan_tag, yuwan_cnt, follownum,
                                rich_tag, loser_tag, cate_length, tag_length, child_length, tag_ratio_1_day_avg,
                                tag_ratio_7_day_avg, tag_ratio_15_day_avg, u_active_mon, u_active_fri, u_active_weekday, u_active_weekend, room_id,
                                anchor_level, anchor_score, is_top200_anchor, room_click_num_1_day, room_click_num_7_day, room_click_num_15_day,
                                room_ctr_1_day, room_ctr_7_day, room_ctr_15_day, i_hot_mon, i_hot_fri, i_hot_weekend, i_hot_weekday, i_ctr_mon,
                                i_ctr_weekend, i_ctr_weekday, guess_cnt_1d, guess_cnt_7d, guess_cnt_15d, raffle_cnt_1d, raffle_cnt_7d, raffle_cnt_15d,
                                room_watch_newmbr_cnt_30d, room_effwatch_mbr_ratio_30d, room_follow_mbr_ratio_30d, room_click_repeat_uids_1d, room_click_repeat_uids_7d,
                                room_click_repeat_uids_15d, click_repeat_avg_nums_15d, item_age, owner_level, item_sex, fans, constellation, click, pay]
    my_label_feature_columns += tag_id_embed
    my_label_feature_columns += cate_id_embed
    my_label_feature_columns += child_id_embed
    # my_label_feature_columns += click
    # my_label_feature_columns += pay
    # my_feature_columns = [uid, room_id, anchor_level, anchor_score,
    #                       is_top200_anchor, room_click_num_1_day, room_click_num_7_day]
    # my_label_feature_columns = [uid, room_id, anchor_level, anchor_score,
    #                             is_top200_anchor, room_click_num_1_day, room_click_num_7_day, click, pay]
#   my_feature_columns += click
#   my_feature_columns += pay
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("feature columns:", my_feature_columns)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return my_feature_columns, my_label_feature_columns


#   # user feature
#   bids = fc.categorical_column_with_hash_bucket("behaviorBids", 10240, dtype=tf.int64)
#   c1ids = fc.categorical_column_with_hash_bucket("behaviorC1ids", 100, dtype=tf.int64)
#   cids = fc.categorical_column_with_hash_bucket("behaviorCids", 10240, dtype=tf.int64)
#   sids = fc.categorical_column_with_hash_bucket("behaviorSids", 10240, dtype=tf.int64)
#   pids = fc.categorical_column_with_hash_bucket("behaviorPids", 1000000, dtype=tf.int64)
#   bids_weighted = fc.weighted_categorical_column(bids, "bidWeights")
#   c1ids_weighted = fc.weighted_categorical_column(c1ids, "c1idWeights")
#   cids_weighted = fc.weighted_categorical_column(cids, "cidWeights")
#   sids_weighted = fc.weighted_categorical_column(sids, "sidWeights")
#   pids_weighted = fc.weighted_categorical_column(pids, "pidWeights")

#   # item feature
#   pid = fc.categorical_column_with_hash_bucket("productId", 1000000, dtype=tf.int64)
#   sid = fc.categorical_column_with_hash_bucket("sellerId", 10240, dtype=tf.int64)
#   bid = fc.categorical_column_with_hash_bucket("brandId", 10240, dtype=tf.int64)
#   c1id = fc.categorical_column_with_hash_bucket("cate1Id", 100, dtype=tf.int64)
#   cid = fc.categorical_column_with_hash_bucket("cateId", 10240, dtype=tf.int64)

#   # context feature
#   matchScore = fc.numeric_column("matchScore", default_value=0.0)
#   popScore = fc.numeric_column("popScore", default_value=0.0)
#   brandPrefer = fc.numeric_column("brandPrefer", default_value=0.0)
#   cate2Prefer = fc.numeric_column("cate2Prefer", default_value=0.0)
#   catePrefer = fc.numeric_column("catePrefer", default_value=0.0)
#   sellerPrefer = fc.numeric_column("sellerPrefer", default_value=0.0)
#   matchType = fc.indicator_column(fc.categorical_column_with_identity("matchType", 9, default_value=0))
#   postition = fc.indicator_column(fc.categorical_column_with_identity("position", 201, default_value=200))
#   triggerNum = fc.indicator_column(fc.categorical_column_with_identity("triggerNum", 51, default_value=50))
#   triggerRank = fc.indicator_column(fc.categorical_column_with_identity("triggerRank", 51, default_value=50))
#   sceneType = fc.indicator_column(fc.categorical_column_with_identity("type", 2, default_value=0))
#   hour = fc.indicator_column(fc.categorical_column_with_identity("hour", 24, default_value=0))
#   phoneBrand = fc.indicator_column(fc.categorical_column_with_hash_bucket("phoneBrand", 1000))
#   phoneResolution = fc.indicator_column(fc.categorical_column_with_hash_bucket("phoneResolution", 500))
#   phoneOs = fc.indicator_column(
#     fc.categorical_column_with_vocabulary_list("phoneOs", ["android", "ios"], default_value=0))
#   tab = fc.indicator_column(fc.categorical_column_with_vocabulary_list("tab",
#         ["ALL", "TongZhuang", "XieBao", "MuYing", "NvZhuang", "MeiZhuang", "JuJia", "MeiShi"], default_value=0))

#   pid_embed = fc.shared_embedding_columns([pids_weighted, pid], 64, combiner='sum', shared_embedding_collection_name="pid")
#   bid_embed = fc.shared_embedding_columns([bids_weighted, bid], 32, combiner='sum', shared_embedding_collection_name="bid")
#   cid_embed = fc.shared_embedding_columns([cids_weighted, cid], 32, combiner='sum', shared_embedding_collection_name="cid")
#   c1id_embed = fc.shared_embedding_columns([c1ids_weighted, c1id], 10, combiner='sum', shared_embedding_collection_name="c1id")
#   sid_embed = fc.shared_embedding_columns([sids_weighted, sid], 32, combiner='sum', shared_embedding_collection_name="sid")
#   # label feature
#   click = fc.numeric_column("click", default_value=0.0)
#   pay = fc.numeric_column("pay", default_value=0.0)
#   global my_feature_columns
#   global my_label_feature_columns
#   my_feature_columns = [matchScore, matchType, postition, triggerNum, triggerRank, sceneType, hour, phoneBrand, phoneResolution,
#              phoneOs, tab, popScore, sellerPrefer, brandPrefer, cate2Prefer, catePrefer]
#   my_feature_columns += pid_embed
#   my_feature_columns += sid_embed
#   my_feature_columns += bid_embed
#   my_feature_columns += cid_embed
#   my_feature_columns += c1id_embed
#   my_label_feature_columns = [matchScore, matchType, postition, triggerNum, triggerRank, sceneType, hour, phoneBrand, phoneResolution,
#              phoneOs, tab, popScore, sellerPrefer, brandPrefer, cate2Prefer, catePrefer, click, pay]
#   my_label_feature_columns += pid_embed
#   my_label_feature_columns += sid_embed
#   my_label_feature_columns += bid_embed
#   my_label_feature_columns += cid_embed
#   my_label_feature_columns += c1id_embed
# #   my_feature_columns += click
# #   my_feature_columns += pay
#   print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#   print("feature columns:", my_feature_columns)
#   print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#   return my_feature_columns, my_label_feature_columns


def parse_exmp(serial_exmp):
    #   click = fc.numeric_column("click", default_value=0, dtype=tf.int64)
    #   pay = fc.numeric_column("pay", default_value=0, dtype=tf.int64)
    #   fea_columns = [click, pay]
    #   fea_columns += my_feature_columns
    fea_columns = my_label_feature_columns
    feature_spec = tf.feature_column.make_parse_example_spec(fea_columns)
    # 把数据映射过来(把真实数据变成上述定义的feature_column的形式)
    feats = tf.parse_single_example(serial_exmp, features=feature_spec)
    click = feats.pop('ctr_label')
    pay = feats.pop('cvr_label')
    return feats, {'ctr': click, 'cvr': pay}
#   return feats, {'ctr': tf.to_float(click), 'cvr': tf.to_float(pay)}


def train_input_fn(filenames, batch_size, shuffle_buffer_size):
    dataset = tf.data.TFRecordDataset(filenames)
    # files = tf.data.Dataset.list_files(filenames)
    # dataset = files.apply(tf.contrib.data.parallel_interleave(
    #     tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))
    # Shuffle, repeat, and batch the examples.
    if shuffle_buffer_size > 0:
        dataset = dataset.shuffle(shuffle_buffer_size)
    #dataset = dataset.apply(tf.contrib.data.map_and_batch(map_func=parse_exmp, batch_size=batch_size))
    #dataset = dataset.repeat().prefetch(1)
    dataset = dataset.map(parse_exmp, num_parallel_calls=8)
    dataset = dataset.repeat().batch(batch_size).prefetch(1)
    print(dataset.output_types)
    print(dataset.output_shapes)
    # Return the read end of the pipeline.
    return dataset


def eval_input_fn(filename, batch_size):
    dataset = tf.data.TFRecordDataset(filename)
    dataset = dataset.map(parse_exmp, num_parallel_calls=8)
    # Shuffle, repeat, and batch the examples.
    dataset = dataset.batch(batch_size)
    # Return the read end of the pipeline.
    return dataset


def build_mode(features, mode, params):
    net = fc.input_layer(features, params['feature_columns'])
    # Build the hidden layers, sized according to the 'hidden_units' param.
    print("!!!!!!!!net {}".format(net))
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
        if 'dropout_rate' in params and params['dropout_rate'] > 0.0:
            net = tf.layers.dropout(net, params['dropout_rate'], training=(
                mode == tf.estimator.ModeKeys.TRAIN))
    # Compute logits
    logits = tf.layers.dense(net, 1, activation=None)
    return logits


def my_model(features, labels, mode, params):  # 特标模参
    with tf.variable_scope('ctr_model'):
        ctr_logits = build_mode(features, mode, params)
    with tf.variable_scope('cvr_model'):
        cvr_logits = build_mode(features, mode, params)

    ctr_predictions = tf.sigmoid(ctr_logits, name="CTR")
    cvr_predictions = tf.sigmoid(cvr_logits, name="CVR")
    prop = tf.multiply(ctr_predictions, cvr_predictions, name="CTCVR")
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'probabilities': prop,
            'ctr_probabilities': ctr_predictions,
            'cvr_probabilities': cvr_predictions
        }
        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    y = labels['cvr']
#     cvr_loss = tf.reduce_sum(
#         tf.keras.backend.binary_crossentropy(y, prop), name="cvr_loss")
    cvr_loss = tf.reduce_sum(
        tf.keras.backend.binary_crossentropy(y, prop), name="cvr_loss")
    ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels['ctr'], logits=ctr_logits), name="ctr_loss")
    loss = tf.add(ctr_loss, cvr_loss, name="ctcvr_loss")

    ctr_accuracy = tf.metrics.accuracy(labels=labels['ctr'], predictions=tf.to_float(
        tf.greater_equal(ctr_predictions, 0.5)))
    cvr_accuracy = tf.metrics.accuracy(
        labels=y, predictions=tf.to_float(tf.greater_equal(prop, 0.5)))
    ctr_auc = tf.metrics.auc(labels['ctr'], ctr_predictions)
    cvr_auc = tf.metrics.auc(y, prop)
    metrics = {'cvr_accuracy': cvr_accuracy, 'ctr_accuracy': ctr_accuracy,
               'ctr_auc': ctr_auc, 'cvr_auc': cvr_auc}
    tf.summary.scalar('ctr_accuracy', ctr_accuracy[1])
    tf.summary.scalar('cvr_accuracy', cvr_accuracy[1])
    tf.summary.scalar('ctr_auc', ctr_auc[1])
    tf.summary.scalar('cvr_auc', cvr_auc[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)
    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    # add debug hook for train
    train_logging_hook = tf.estimator.LoggingTensorHook(
        {'ctr_labels': labels['ctr'], 'ctr_predictions': ctr_logits, 'cvr_labels': labels['cvr'], 'cvr_predictions': prop, 'cvr_loss': cvr_loss, 'ctr_loss': ctr_loss}, every_n_iter=10)

    # optimizer = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
    # train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    train_op = tf.train.AdamOptimizer(params['learning_rate']).minimize(
            loss, global_step=tf.train.get_or_create_global_step())
    # return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op, training_hooks=[train_logging_hook])
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(unused_argv):
    set_tfconfig_environ()
    create_feature_columns()
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,  # 特征列变量
            'hidden_units': FLAGS.hidden_units.split(','),
            'learning_rate': FLAGS.learning_rate,
            'dropout_rate': FLAGS.dropout_rate
        },
        config=tf.estimator.RunConfig(
            model_dir=FLAGS.model_dir, save_checkpoints_steps=FLAGS.save_checkpoints_steps)
    )
    batch_size = FLAGS.batch_size
    print("train steps:", FLAGS.train_steps, "batch_size:", batch_size)
    if isinstance(FLAGS.train_data, str) and os.path.isdir(FLAGS.train_data):
        train_files = [FLAGS.train_data + '/' + x for x in os.listdir(FLAGS.train_data)] if os.path.isdir(
            FLAGS.train_data) else FLAGS.train_data
    else:
        train_files = FLAGS.train_data
    if isinstance(FLAGS.eval_data, str) and os.path.isdir(FLAGS.eval_data):
        eval_files = [FLAGS.eval_data + '/' + x for x in os.listdir(FLAGS.eval_data)] if os.path.isdir(
            FLAGS.eval_data) else FLAGS.eval_data
    else:
        eval_files = FLAGS.eval_data
    shuffle_buffer_size = FLAGS.shuffle_buffer_size
    train_files_records = 0
    for tff in train_files:
        for record in tf.python_io.tf_record_iterator(tff):
            train_files_records += 1
    print("{} train_data: {}".format(train_files_records, train_files))
    print("eval_data:", eval_files)
    print("shuffle_buffer_size:", shuffle_buffer_size)
    train_steps = FLAGS.train_epochs * train_files_records/FLAGS.batch_size

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(
            train_files, batch_size, shuffle_buffer_size),
        max_steps=train_steps
    )
    def input_fn_for_eval(): return eval_input_fn(eval_files, batch_size)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=input_fn_for_eval, throttle_secs=600, steps=None)

    print("before train and evaluate")
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print("after train and evaluate")

    # Evaluate accuracy.
    results = classifier.evaluate(input_fn=input_fn_for_eval)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))
    print("after evaluate")

    if FLAGS.job_name == "worker" and FLAGS.task_index == 0:
        print("exporting model ...")
        feature_spec = tf.feature_column.make_parse_example_spec(
            my_feature_columns)
        print(feature_spec)
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            feature_spec)
        classifier.export_savedmodel(
            FLAGS.output_model, serving_input_receiver_fn)
    print("quit main")


if __name__ == "__main__":
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print("CUDA_VISIBLE_DEVICES:", os.environ["CUDA_VISIBLE_DEVICES"])
    if FLAGS.run_on_cluster:
        parse_argument()
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
