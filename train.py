# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BILSTM-CRF
# @File     :train
# @Date     :2021/7/11 23:07
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
from models.spanNER import SpanNER
from config.config import Config
from data_load.dataloader import MyDataset,collate_fn
from Framework.framework import Framework
from torch.utils.data import DataLoader
import torch
torch.manual_seed(1234)
con = Config()
train_file = "dataset/train_data.json"
val_file = "dataset/val_data.json"

train_dataset = MyDataset(con,train_file)
train_dataload = DataLoader(train_dataset,shuffle=True,batch_size=con.batch_size,
                            pin_memory=True,collate_fn=collate_fn)

val_dataset = MyDataset(con,val_file,flag="val")
val_dataload = DataLoader(val_dataset,batch_size=1,pin_memory=True,collate_fn=collate_fn)

model = SpanNER(con)
fw = Framework(con)
fw.train(model,train_dataload,val_dataload)



