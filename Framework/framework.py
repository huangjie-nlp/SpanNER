import datetime
from models.BertSpanNER import Model
from dataloader.dataloader import MyDataset,collate_fn
import torch
import torch.nn.functional as F
import json
from torch.utils.data import DataLoader
from tqdm import tqdm
from logger.Logger import Logger
import numpy as np


class Framework():
    def __init__(self,config):
        self.config = config
        self.logger = Logger(self.config.log.format(datetime.datetime.now().strftime("%Y-%m-%d - %H:%M:%S")))
        with open(self.config.type2id,"r",encoding="utf-8") as f:
            self.id2type = json.load(f)[1]
        self.device = torch.device("cuda:%d"%self.config.cuda if torch.cuda.is_available() else "cpu")

    def train(self):

        def loss_fn(pred,target,mask):
            """
            :param pred:shape:[batch,seq_length,type_num]
            :param target: [batch,seq_length,type_num]
            :param mask: [batch,seq_length]
            :return: tensor
            """
            mask = mask.unsqueeze(dim=-1)
            los = F.binary_cross_entropy(pred,target)
            loss = torch.sum(los * mask) / torch.sum(mask)
            return loss

        train_dataset = MyDataset(self.config,self.config.train_fn)
        dev_dataset = MyDataset(self.config,self.config.dev_fn)
        train_dataloader = DataLoader(train_dataset,shuffle = True,batch_size = self.config.batch_size,
                                      pin_memory = True,collate_fn = collate_fn)
        dev_dataloader = DataLoader(dev_dataset,batch_size = 1,
                                      pin_memory = True,collate_fn = collate_fn)

        model = Model(self.config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=self.config.learning_rate)

        best_f1,precision,recall,best_epoch = 0,0,0,0
        global_step = 0
        global_loss = 0
        for epoch in range(1,self.config.epoch+1):
            print("[{}/{}]".format(epoch,self.config.epoch))
            for data in tqdm(train_dataloader):
                head,tail = model(data)
                head_loss = loss_fn(head,data["head"].to(self.device),data["mask"].to(self.device))
                tail_loss = loss_fn(tail,data["tail"].to(self.device),data["mask"].to(self.device))
                loss = head_loss + tail_loss

                optimizer.zero_grad()
                loss.backward()
                global_loss += loss.item()
                optimizer.step()

                if (global_step + 1) % self.config.step == 0:
                    self.logger.logger.info("epoch:{} global_step:{} global_loss:{:5.4f}".
                                            format(epoch,global_step,global_loss))
                    global_loss = 0
                global_step += 1
            p, r, f1_score, predict = self.evaluate(model,dev_dataloader,self.config.h_bar,self.config.t_bar)
            if f1_score > best_f1:
                best_f1 = f1_score
                precision = p
                recall = r
                best_epoch = epoch
                print("epoch {} save model......".format(epoch))
                torch.save(model.state_dict(),self.config.save_model)
                json.dump(predict,open(self.config.dev_result,"w",encoding="utf-8"),
                          indent=4,ensure_ascii=False)
            self.logger.logger.info("precision:{:5.4f} recall:{:5.4f} f1_score:{:5.4f} "
                                    "best_f1_score:{:5.4f} best_epoch:{}".
                                    format(precision,recall,f1_score,best_f1,best_epoch))


    def evaluate(self,model,dataloader,h_bar=0.5,t_bar=0.5):
        model.eval()

        predict_num,gold_num,correct_num = 0,0,0
        predict = []

        with torch.no_grad():
            for data in tqdm(dataloader):
                start,end = model(data)
                head_idx,tail_idx = np.where(start.cpu()[0] > h_bar),np.where(end.cpu()[0] > t_bar)
                token = data["token"][0]
                resl = []
                for h_idx,h_type in zip(*head_idx):
                    for t_idx,t_type in zip(*tail_idx):
                        if t_idx >= h_idx and h_type == t_type:
                            entity = "".join(token[h_idx:t_idx+1])
                            idx_type = str(h_idx-1)+"/"+str(t_idx-1)+"/"+self.id2type[str(t_type)]
                            resl.append(idx_type+"@"+entity)
                            break
                predict.append({"sentence":data["sentence"][0],"gold":data["entity_list"][0],"predict":resl,
                                "new":list(set(resl)-set(data["entity_list"][0])),
                                "lack":list(set(data["entity_list"][0])-set(resl))})
                predict_num += len(set(resl))
                gold_num += len(set(data["entity_list"][0]))
                correct_num += len(set(resl) & set(data["entity_list"][0]))
        print("predict_num:{} gold_num:{} correct_num:{}".format(predict_num,gold_num,correct_num))
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        return precision,recall,f1_score,predict

    def test(self):
        model = Model(self.config)
        model.load_state_dict(torch.load(self.config.save_model,map_location=self.device))
        model.to(self.device)
        model.eval()

        dataset = MyDataset(self.config,self.config.test_fn)
        dataloader = DataLoader(dataset,shuffle=True,batch_size=1,
                                collate_fn=collate_fn,pin_memory=True)

        predict_num,gold_num,correct_num = 0,0,0
        predict = []

        with torch.no_grad():
            for data in tqdm(dataloader):
                start,end = model(data)
                head_idx,tail_idx = np.where(start.cpu()[0] > self.config.h_bar),\
                                    np.where(end.cpu()[0] > self.config.t_bar)
                token = data["token"][0]
                resl = []
                for h_idx,h_type in zip(*head_idx):
                    for t_idx,t_type in zip(*tail_idx):
                        if t_idx >= h_idx and t_type == h_type:
                            entity = "".join(token[h_idx:t_idx+1])
                            idx_type = str(h_idx-1)+"/"+str(t_idx-1)+"/"+self.id2type[str(t_type)]
                            resl.append(idx_type+"@"+entity)
                            break
                predict.append(
                    {"sentence": data["sentence"][0], "gold": data["entity_list"][0], "predict": resl,
                     "new": list(set(resl) - set(data["entity_list"][0])),
                     "lack": list(set(data["entity_list"][0]) - set(resl))})
                predict_num += len(set(resl))
                gold_num += len(set(data["entity_list"][0]))
                correct_num += len(set(resl) & set(data["entity_list"][0]))
        print("predict_num:{} gold_num:{} correct_num:{}".format(predict_num, gold_num, correct_num))
        precision = correct_num / (predict_num + 1e-10)
        recall = correct_num / (gold_num + 1e-10)
        f1_score = 2 * precision * recall / (precision + recall + 1e-10)
        return precision, recall, f1_score, predict
