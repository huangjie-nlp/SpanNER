import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)
        self.start = nn.Linear(self.config.bert_dim,self.config.type_num)
        self.end = nn.Linear(self.config.bert_dim,self.config.type_num)
        self.dropout = nn.Dropout(self.config.dropout)
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")

    def forward(self,data):
        input_ids = data["input_ids"].to(self.device)
        mask = data["mask"].to(self.device)
        # [batch_size,seq_length,bert_dim]
        encode = self.bert(input_ids,attention_mask=mask)[0]
        start = self.start(encode)
        end = self.end(encode)
        return F.sigmoid(start),F.sigmoid(end)