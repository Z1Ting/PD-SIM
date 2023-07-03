# from simclr_new import SimCLR_medical
# from simclr_new2 import SimCLR_medical
from PD-SIM import SimCLR_medical
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




def main():
    mid = open("G:/Experience/config.yaml", encoding='utf-8')
    config = yaml.load(mid, Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    simclr = SimCLR_medical(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
