# -*- coding: utf-8 -*-
"""
-------------------------------------------------
# @Project  :BILSTM-CRF
# @File     :parse_data
# @Date     :2021/7/11 14:03
# @Author   :huangjie
# @Email    :728155808@qq.com
# @Software :PyCharm
-------------------------------------------------
"""
import json
def read_data(fn):
    data = []
    with open(fn,"r",encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            temp = line.split("|||")[:-1]
            inject_sample = {"text":temp[0],"entity":[]}
            for i in temp[1:]:
                entity_list = i.split("    ")
                inject_sample["entity"].append(
                    entity_list[0]+"/"+entity_list[1]+"/"+entity_list[2]+"@"+temp[0][int(entity_list[0]):int(entity_list[1])+1])
            data.append(inject_sample)
    json.dump(data,open("../dataset/train_data.json","w",encoding="utf-8"),indent=4,ensure_ascii=False)
file = "../data/train_data.txt"
read_data(file)