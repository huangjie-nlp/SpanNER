# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BILSTM-CRF
# @File     :config
# @Date     :2021/7/11 14:30
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""

class Config():
    def __init__(self):
        self.max_len = 300
        self.vocab_size = 2839
        self.type2id = "dataset/type2id.json"
        self.vocab = "dataset/vocab.json"
        self.word_dim = 128
        self.lstm = 128
        self.hidden = 128
        self.num_type = 9
        self.lr = 5e-4
        self.epoch = 200
        self.eval_log = "eval_log/val_log.json"
        self.batch_size = 256
