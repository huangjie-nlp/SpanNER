# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BILSTM-CRF
# @File     :data_processing
# @Date     :2021/7/11 14:18
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import json

def get_entity_type(train_fn):
    type2id = {}
    id2type = {}
    with open(train_fn,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split("|||")[:-1]
            for i in temp[1:]:
                entity_list = i.split("    ")
                if entity_list[-1] not in type2id:
                    type2id[entity_list[-1]] = len(type2id)
                    id2type[len(id2type)] = entity_list[-1]
    return type2id,id2type

file = "../data/train_data.txt"
type2id,id2type = get_entity_type(file)
json.dump([type2id,id2type],open("../dataset/type2id.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

def generate_vocab(train_fn):
    data = json.load(open(train_fn,"r",encoding="utf-8"))
    vocab = {"pad":0,"unk":1}
    for i in data:
        for j in i["text"]:
            if j not in vocab:
                vocab[j] = len(vocab)
    return vocab
train_file = "../dataset/train_data.json"
data = generate_vocab(train_file)
json.dump(data,open("../dataset/vocab.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)

