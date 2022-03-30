
class Config():
    def __init__(self):
        self.bert_path = "./bert-base-chinese"
        self.bert_dim = 768
        self.type2id = "./dataset/type2id.json"
        self.type_num = 9
        self.train_fn = "./dataset/train_data.json"
        self.dev_fn = "./dataset/dev_data.json"
        self.test_fn = "./dataset/dev_data.json"
        self.dropout = 0.5
        self.cuda = 0
        self.batch_size = 16
        self.epoch = 10
        self.learning_rate = 1e-5
        self.step = 300
        self.log = "log/{}_log.log"
        self.h_bar = 0.5
        self.t_bar = 0.5
        self.save_model = "checkpoint/BertSpanNER.pt"
        self.dev_result = "dev_result/dev.json"
        self.test_result = "test_result/test.json"
