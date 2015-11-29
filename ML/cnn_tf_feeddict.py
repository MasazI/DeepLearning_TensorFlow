#encoding: utf-8

import tensorflow as tf

def fill_feed_dict(data_set, images_pl, labels_pl, batch_size, fake_data):
    '''
    学習のstepに必要なfeed_dictを生成する
    '''
    images_feed, labels_feed = data_set.next_batch(batch_size, fake_data)

    # key: placeholder、value: feed(次のステップに必要なデータ)
    feed_dict = {images_pl: images_feed, labels_pl: labels_feed,}

    return feed_dict
