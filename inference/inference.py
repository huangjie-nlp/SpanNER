
import torch
from models.BertSpanNER import Model
from transformers import BertTokenizer
import numpy as np
import json
from config.config import Config

class Inference():
    def __init__(self,config):
        self.config = config
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")
        self.model = Model(self.config)
        self.model.load_state_dict(torch.load(self.config.save_model,map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        self.id2type = json.load(open(self.config.type2id, "r", encoding="utf-8"))[1]

    def __data_processing(self,sentence):
        token = ["[CLS]"] + list(sentence) + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(token)
        mask = [1]*len(token)
        input_ids = torch.LongTensor([input_ids])
        mask = torch.LongTensor([mask])
        return {"input_ids":input_ids,
                "mask":mask,
                "token":token}

    def predict(self,sentence):
        data = self.__data_processing(sentence)
        with torch.no_grad():
            start,end = self.model(data)
            head_idx,tail_idx = np.where(start.cpu()[0] > self.config.h_bar),np.where(end.cpu()[0] > self.config.t_bar)
            resl = []
            token = data["token"]
            for h_idx,h_type in zip(*head_idx):
                for t_idx,t_type in zip(*tail_idx):
                    if t_idx >= h_idx and h_type == t_type:
                        entity = "".join(token[h_idx:t_idx + 1])
                        idx_type = str(h_idx - 1) + "/" + str(t_idx - 1) + "/" + self.id2type[str(t_type)]
                        resl.append(idx_type + "@" + entity)
                        break
        print(json.dumps({"sentence":sentence,"predict":resl},indent=4,ensure_ascii=False))
