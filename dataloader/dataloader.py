import torch
from torch.utils.data import Dataset
import numpy as np
import json
from transformers import BertTokenizer

class MyDataset(Dataset):
    def __init__(self,config,fn):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path)
        with open(fn,"r",encoding="utf-8") as f:
            self.data = json.load(f)
        with open(self.config.type2id,"r",encoding="utf-8") as type_f:
            self.type2id = json.load(type_f)[0]
        self.data = [data for data in self.data if len(data["text"]) < 510]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ins_json_data = self.data[idx]
        sentence = ins_json_data["text"]
        entity_list = ins_json_data["entity"]
        token =["[CLS]"] + list(sentence) + ["[SEP]"]
        token2id = self.tokenizer.convert_tokens_to_ids(token)
        token_len = len(token2id)
        if token_len > 512:
            print("sentence:",sentence)
            print("token_len:",token_len)
        assert token_len <= 512
        mask = [1] * token_len

        input_ids = np.array(token2id)
        mask = np.array(mask)

        head,tail = np.zeros((token_len,self.config.type_num)),np.zeros((token_len,self.config.type_num))
        for e in entity_list:
            type_idx,entity = e.split("@")
            start,end,e_type = type_idx.split("/")
            head[int(start) + 1][self.type2id[e_type]] = 1
            tail[int(end) + 1][self.type2id[e_type]] = 1

        return token,input_ids,mask,head,tail,sentence,entity_list,token_len

def collate_fn(batch):
    token, input_ids, mask, head, tail, sentence, entity_list, token_len = zip(*batch)

    cur_batch = len(batch)
    max_len = max(token_len)

    batch_input_ids = torch.LongTensor(cur_batch,max_len).zero_()
    batch_mask = torch.LongTensor(cur_batch,max_len).zero_()
    batch_head = torch.Tensor(cur_batch,max_len,9).zero_()
    batch_tail = torch.Tensor(cur_batch,max_len,9).zero_()

    for i in range(cur_batch):
        batch_input_ids[i,:token_len[i]].copy_(torch.from_numpy(input_ids[i]))
        batch_mask[i,:token_len[i]].copy_(torch.from_numpy(mask[i]))
        batch_head[i,:token_len[i]].copy_(torch.from_numpy(head[i]))
        batch_tail[i,:token_len[i]].copy_(torch.from_numpy(tail[i]))

    return {"input_ids":batch_input_ids,
            "mask":batch_mask,
            "head":batch_head,
            "tail":batch_tail,
            "token":token,
            "sentence":sentence,
            "entity_list":entity_list}

if __name__ == '__main__':
    from config.config import Config
    from torch.utils.data import DataLoader
    config = Config()
    dataset = MyDataset(config,config.dev_fn)
    dataloader = DataLoader(dataset,batch_size=1,collate_fn=collate_fn)
    for data in dataloader:
        print(data)
