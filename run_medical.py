
from simclr_new3 import SimCLR_medical
import yaml
from data_aug.dataset_wrapper import DataSetWrapper
import ssl
import torch
from torch import random
import os
import numpy as np
import time
ssl._create_default_https_context = ssl._create_unverified_context




def main():
   
    mid = open("config.yaml", "r")

   
    config = yaml.load(mid, Loader=yaml.FullLoader)

    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    simclr = SimCLR_medical(dataset, config)
    simclr.train()








if __name__ == "__main__":
    main()
