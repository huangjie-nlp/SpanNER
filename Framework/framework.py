# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BILSTM-CRF
# @File     :framework
# @Date     :2021/7/11 15:49
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import torch
import torch.nn.functional as F
import json
import numpy as np
import time
torch.manual_seed(1234)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Framework():
    def __init__(self,con):
        self.con = con

    def train(self,model,train_dataloader,val_dataloader):
        model.to(device)
        def loss_fn(targ,pred,mask):
            mask = mask.unsqueeze(dim=-1)
            los = F.binary_cross_entropy(pred,targ)
            loss = torch.sum(los * mask) / torch.sum(mask)
            return loss

        optimizer = torch.optim.AdamW(model.parameters(),self.con.lr)
        best_f1 = -1
        best_epoch = -1

        for i in range(self.con.epoch):
            start_time = time.time()
            total_loss = 0
            for data in train_dataloader:
                token = data["token"].to(device)
                mask = data["mask"].to(device)
                start,end = model(token)
                head_loss = loss_fn(data["entity_head"].to(device),start,mask)
                tail_loss = loss_fn(data["entity_tail"].to(device),end,mask)
                loss = head_loss + tail_loss

                model.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print("epoch %d loss %5.2f"%(i,total_loss))
            if i % 5 == 0:
                f1_score, precision, recall = self.evaluate(model,val_dataloader)
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_epoch = i
                print("f1_score:%5.2f, precision:%5.2f, recall:%5.2f,best_f1:%5.2f,best_epoch:%d"%(f1_score,precision,recall,best_f1,best_epoch))

    def evaluate(self,model,dataloader,h_bar=0.5,t_bar=0.5):
        id2type = json.load(open(self.con.type2id,"r",encoding="utf-8"))[1]
        model.eval()
        predict = []
        gold_num = 0
        predict_num = 0
        correct_num = 0
        with torch.no_grad():
            for i in dataloader:
                entity_predict = []
                sentence = i["sentence"][0]
                entity = i["entity"][0]
                data = i["token"].to(device)
                start,end = model(data)
                head_predict,tail_predict = np.where(start.cpu()[0] > h_bar),np.where(end.cpu()[0] > t_bar)
                for h_idx,e_type in zip(*head_predict):
                    for t_idx,te_type in zip(*tail_predict):
                        if t_idx >= h_idx and e_type==te_type:
                            res = str(h_idx)+"/"+str(t_idx)+"/"+id2type[str(e_type)]+"@"+"".join(sentence[h_idx:t_idx+1])
                            entity_predict.append(res)
                            break
                gold_num += len(entity)
                predict_num += len(entity_predict)
                correct_num += len(set(entity) & set(entity_predict))
                predict.append({"text":"".join(sentence),
                                "gold":entity,
                                "predict":entity_predict,
                                "new":list(set(entity_predict)-set(entity)),
                                "lack":list(set(entity)-set(entity_predict))})
        json.dump(predict,open(self.con.eval_log,"w",encoding="utf-8"),indent=4,ensure_ascii=False)
        precision = correct_num / (predict_num+1e-10)
        recall = correct_num / (gold_num+1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        model.train()
        return f1_score,precision,recall