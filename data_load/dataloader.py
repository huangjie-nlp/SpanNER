# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BILSTM-CRF
# @File     :dataloader
# @Date     :2021/7/11 14:36
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
from torch.utils.data import Dataset,DataLoader
import numpy as np
import torch
import json

class MyDataset(Dataset):
    def __init__(self,con,fn,flag="train"):
        self.con = con
        self.data = json.load(open(fn,"r",encoding="utf-8"))
        if flag == "train":
            self.data = [i for i in self.data if len(i["text"]) < 300]
        self.maxlen = self.con.max_len
        self.type2id = json.load(open(self.con.type2id,"r",encoding="utf-8"))[0]
        self.vocab = json.load(open(self.con.vocab,"r",encoding="utf-8"))
        self.flag = flag
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        sentence = list(sample["text"])
        entity_list = sample["entity"]
        sentence2id = []
        token_len = len(sentence)
        for i in sentence:
            if i in self.vocab:
                sentence2id.append(self.vocab[i])
            else:
                sentence2id.append(self.vocab["unk"])
        sentence2id = np.array(sentence2id)
        mask = np.ones(token_len)
        entity_head = np.zeros((token_len,len(self.type2id)))
        entity_tail = np.zeros((token_len,len(self.type2id)))
        for i in entity_list:
            split_entity = i.split("@")[0]
            head ,tail,e_type = split_entity.split("/")
            entity_head[int(head)][self.type2id[e_type]] = 1
            entity_tail[int(tail)][self.type2id[e_type]] = 1

        return token_len,sentence,entity_list,sentence2id,mask,entity_head,entity_tail

def collate_fn(batch):
    batch = list(filter(lambda x:x is not None,batch))
    token_len, sentence, entity_list, sentence2id, mask, entity_head, entity_tail = zip(*batch)
    cur_batch = len(batch)
    batch_max_len = max(token_len)

    batch_sentence = torch.zeros(cur_batch,batch_max_len).long()
    batch_mask = torch.zeros(cur_batch,batch_max_len).long()
    batch_entity_head = torch.zeros(cur_batch,batch_max_len,9)
    batch_entity_tail = torch.zeros(cur_batch,batch_max_len,9)

    for i in range(cur_batch):
        batch_sentence[i,:token_len[i]].copy_(torch.from_numpy(sentence2id[i]))
        batch_mask[i,:token_len[i]].copy_(torch.from_numpy(mask[i]))
        batch_entity_head[i,:token_len[i],:].copy_(torch.from_numpy(entity_head[i]))
        batch_entity_tail[i,:token_len[i],:].copy_(torch.from_numpy(entity_tail[i]))

    return {"sentence":sentence,
            "entity":entity_list,
            "token":batch_sentence,
            "mask":batch_mask,
            "entity_head":batch_entity_head,
            "entity_tail":batch_entity_tail}

if __name__ == '__main__':
    data_fn = "../dataset/val_data.json"
    from config.config import Config
    con = Config()
    dataset = MyDataset(con,data_fn,flag="val")
    dataloader = DataLoader(dataset,batch_size=1,collate_fn=collate_fn)
    for i in dataloader:
        print(i)
