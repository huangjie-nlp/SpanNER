# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BILSTM-CRF
# @File     :spanNER
# @Date     :2021/7/11 15:39
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch.nn as nn
import torch
torch.manual_seed(1234)

class SpanNER(nn.Module):
    def __init__(self,con):
        super(SpanNER, self).__init__()
        self.con = con
        self.embeding = nn.Embedding(self.con.vocab_size,self.con.word_dim)
        self.bilstm = nn.LSTM(self.con.word_dim,self.con.lstm,num_layers=1,bidirectional=True,batch_first=True)
        self.Linear = nn.Linear(2*self.con.lstm,self.con.hidden)
        self.start = nn.Linear(self.con.hidden,self.con.num_type)
        self.end = nn.Linear(self.con.hidden,self.con.num_type)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self,data):
        # [batch_size,seq_len,word_dim(128)]
        emb = self.embeding(data)
        # [batch_size,seq_len,256]
        bilstm,(cell,hidden) = self.bilstm(emb)
        # [batch_size, seq_len, 128]
        linear = self.Linear(bilstm)
        # [batch_size,seq_len,num_type]
        start = self.start(linear)
        # [batch_size,seq_len,num_type]
        end = self.end(linear)
        return torch.sigmoid(start),torch.sigmoid(end)

