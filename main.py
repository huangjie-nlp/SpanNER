
from framework.Framework import Framework
from config.config import Config
import torch
import numpy as np
import json

config = Config()
fw = Framework(config)
print("*"*50+"train"+"*"*50)
fw.train()

print("*"*50+"test"+"*"*50)
precision, recall, f1_score, predict = fw.test()
json.dump(predict,open(config.test_result,"w",encoding="utf-8"),indent=4,ensure_ascii=False)
print("precision:{:5.4f} recall:{:5.4f} f1_score:{:5.4f}".format(precision,recall,f1_score))
