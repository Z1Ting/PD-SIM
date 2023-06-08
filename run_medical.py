# from simclr_new import SimCLR_medical
# from simclr_new2 import SimCLR_medical
from simclr_new3 import SimCLR_medical
# from simclr_new4 import SimCLR_medical
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import ssl
import torch
from torch import random
import os
import numpy as np
import time
ssl._create_default_https_context = ssl._create_unverified_context

# seed = 2023
# random.seed()
# os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# print('ramdon_seed is success')


def main():
    # open()函数打开文件并返回相应的文件对象。打开文件进行读取。（默认
    # mid = open("config.yaml", "r")
    mid = open("G:/Experience/SimCLR-pytorch-master/config.yaml", encoding='utf-8')

    # python通过open方式读取文件数据，再通过load函数将数据转化为列表或字典；FullLoader - -加载完整的YAML语言。避免任意代码执行。这是当前（PyYAML5.1）默认加载器调用
    # yaml.load(input)（发出警告后）。
    config = yaml.load(mid, Loader=yaml.FullLoader)
    # print(config)
    # return train_loader, valid_loader
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    # print(config['dataset'])
    simclr = SimCLR_medical(dataset, config)
    simclr.train()








if __name__ == "__main__":
    main()
