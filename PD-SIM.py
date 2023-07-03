import torch
import torch.nn as nn
import  torch.nn.functional as F

import time
from models.resnet_simclr import ResNetSimCLR
from models.resnet_simclr_3D_recombiantion import ResNetSimCLR_3D

from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from loss.nt_xent import NTXentLoss
from loss.margin_triplet import MarginTripletLoss
from loss.nt_logistic import NTLogisticLoss
from LARS import LARS
import os
import shutil
import sys


from Data_Loader4_Recombination import data_flow4_Recombination

from Data_Loader3 import tst_data_flow
import scipy.io as sio
import h5py

import yaml





apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False

import numpy as np

torch.manual_seed(0)

start_time = time.time()


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('G:\Experience\config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

mid = open("G:\Experience\config.yaml", encoding='utf-8')
config = yaml.load(mid, Loader=yaml.FullLoader)

task = config['task']
epoch = config['epochs']
batch_size = config['batch_size']
patch_size = config['patch_size']
patch_num = config['patch_num']
patch_num_1 = config['patch_num_1']
select_patch_num = config['select_patch_num']  # new
kernel_num = [32, 64, 128, 128, batch_size]
img_path = "G:\Remote Data Synchronizing Folders\predMRIdata/newADNI3/"


data_flow = data_flow4_Recombination


if task == 1:
    task_name = 'AD_classification'
elif task == 2:
    task_name = 'CN_EMCI'
elif task == 3:
    task_name = 'MCI_conversion'
elif task == 4:
    task_name = 'CN_LMCI'
print(task_name)
data = sio.loadmat('G:/data_split/{}/{}.mat'.format(task_name, config['datalist']))  #读取.mat文件
sample_name = data['samples_train'].flatten()  #
labels = data['labels_train'].flatten()  #

sample_valid_list = np.squeeze(data['sample_valid_list'])
sample_train_list = np.squeeze(data['sample_train_list'])
res = np.zeros(shape=(5, 4))

valid_list = np.squeeze(sample_valid_list[0])
train_list = np.squeeze(sample_train_list[0])


labels_train = labels[train_list]
labels_valid = labels[valid_list]
samples_train = sample_name[train_list]
samples_valid = sample_name[valid_list]

template_cors = h5py.File('G:\PearsonCorrCoef_test\PatchSelectResult/'
                              'template_center_size{}__num{}.mat'.format(patch_size, patch_num), 'r')

template_cors = template_cors['patch_centers']



use_amp = False
if use_amp:
    scaler = torch.cuda.amp.GradScaler()



class SimCLR_medical(object):

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter()
        self.dataset = dataset
        self.batch_size = config['batch_size']
        self.model = ResNetSimCLR_3D(**self.config["model"]).to(self.device)
        self.loss_func = self._choose_loss()

    def _choose_loss(self):
        if self.config['loss_select'] == 'NT_Xent':
            print("using NT_Xent as loss func")
            return NTXentLoss(self.device, self.config['batch_size'], **self.config['loss'])
        elif self.config['loss_select'] == 'NT_Logistic':
            print("using NT_Logistic as loss func")
            return NTLogisticLoss(self.device, self.config['batch_size'], **self.config['loss'])
        elif self.config['loss_select'] == 'MarginTriplet':
            print("using MarginTriplet as loss func")
            return MarginTripletLoss(self.device, self.config['batch_size'], self.config['semi_hard'],
                                     **self.config['loss'])
        else:
            print('not a valid loss, use NT_Xent as default')
            return NTXentLoss(self.device, self.config['batch_size'], **self.config['loss'])

    def _get_device(self):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return device

    def _step(self, model, xis, xjs, n_iter):


        ris, zis = model(xis)  # [N,C]

        rjs, zjs = model(xjs)  # [N,C]
        '''input_shape:
        torch.Size([10, 1, 2, 2, 2])
        h_shape:
        torch.Size([10, 512])
        x_shape:
        torch.Size([10, 128])
        '''
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)
        if use_amp:
            with torch.cuda.amp.autocast():
                loss = self.loss_func(zis, zjs)
        else:
            loss = self.loss_func(zis, zjs)


        return loss
    # 5 fold
for i in range(5):
    # 20% training samples as the validation set
    valid_list = np.squeeze(sample_valid_list[i])
    train_list = np.squeeze(sample_train_list[i])

    labels_train = labels[train_list]
    labels_valid = labels[valid_list]
    samples_train = sample_name[train_list]
    samples_valid = sample_name[valid_list]


    def train(self):

        train_loader = data_flow(img_path, samples_train, labels_train, template_cors,
                                 batch_size, patch_size,
                                 patch_num,index_list_1,index_list_2)
        valid_loader = data_flow(img_path, samples_valid, labels_valid, template_cors,
                                 batch_size, patch_size,
                                 patch_num,index_list_1,index_list_2)

        model = self.model


        if self.config['optimizer'] == 'LARS':
            print('using LARS as optimizer.')
            optimizer = LARS(model.parameters(), lr=0.3 * self.batch_size / 256, eta=1e-3,
                             weight_decay=eval(self.config['weight_decay']))
        elif self.config['optimizer'] == 'SGD':
            print('using SGD as optimizer. In order to obtain less space')
            optimizer = torch.optim.SGD(model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))
        else:
            print('using Adam as optimizer.')
            optimizer = torch.optim.Adam(model.parameters(), lr=eval(self.config['learning_rate']), weight_decay=eval(self.config['weight_decay']))

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000000001,
                                                               last_epoch=-1)

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(model, optimizer,
                                              opt_level='O2',
                                              keep_batchnorm_fp32=True)
        model_checkpoints_folder = os.path.join(self.writer.log_dir, task_name, 'checkpoints')

        _save_config_file(model_checkpoints_folder)

        n_iter = 0
        l = []
        train_a = []
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_epoch = 0
        best_acc = 0

        if

        for epoch_counter in range(self.config['epochs']):
            start_time = time.time()
            print("now in epoch {0}".format(epoch_counter))

            for i_batch in range(len(train_list) // batch_size):
                xis, xjs, outputs = next(
                    train_loader)

                optimizer.zero_grad()
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, n_iter)


                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    loss.backward()
                optimizer.step()

                n_iter += 1
            with torch.no_grad():
                start_time = time.time()
                if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                    self.device = self._get_device()
                    valid_loss = self._validate(model, valid_loader, valid_list, batch_size, epoch_counter)
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        best_epoch = epoch_counter
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'recombination+model1+nxtent+olVALID_{}fold.pth'.format(i+1)))

                    self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                    valid_n_iter += 1
                    print('[NO.{} epoch best_valid_loss is:{}]'.format(best_epoch, best_valid_loss))


            if epoch_counter+1 >= 3:  # 10
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)
            end_time = time.time()
            print('In epoch {0}, valid time cost:{1}'.format(epoch_counter, end_time - start_time))
            print('--------------------------------------------')



    def _validate(self, model, valid_loader, valid_list, batch_size, epoch_counter):
        with torch.no_grad():
            model.eval()

            a = []
            VALID_ACC = []
            fina_acc = []
            valid_loss = 0.0
            l = []
            counter = 0
            for i_batch in range(len(valid_list) // batch_size):
                xis, xjs, outputs = next(valid_loader)

                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs, counter)

                valid_loss += loss.item()
                counter += 1

                l.append(loss.item())

                print('epoch{} batch{} VALID]  '  
                      ' cur_loss:{:.4f} mean_loss:{:.4f}'.format(epoch_counter, i_batch,
                                                                 l[-1],
                                                                 np.mean(l)))

            valid_loss /= counter
        model.train()
        return valid_loss


