from inference.inference import Inference
from config.config import Config

config = Config()
inference = Inference(config)
if __name__ == '__main__':
    while True:
        sentence = input("句子:")
        inference.predict(sentence)