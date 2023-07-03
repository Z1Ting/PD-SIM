import os
import numpy as np
import scipy.io as sio
from random import sample
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

task = 1
img_path = "G:\Remote Data Synchronizing Folders\predMRIdata\predCuingnet/"
sample_name = []
labels = []

if task == 1:
    for img in os.listdir(img_path + 'AD/mwp1'):  # F:\Remote Data Synchronizing Folders\predMRIdata\newADNI3\AD\mwp1
        img_fpath = img_path + "AD/mwp1/" + img
        if os.path.isfile(img_fpath) and img.split(".")[-1] == "nii":
            sample_name.append('AD/mwp1/' + img)
            labels.append(1)
            # print('AD/mwp1/' + img)
    for img in os.listdir(img_path + 'CN/mwp1'):
        img_fpath = img_path + "CN/mwp1/" + img
        if os.path.isfile(img_fpath) and img.split(".")[-1] == "nii":
            sample_name.append('CN/mwp1/' + img)
            labels.append(0)
    task_name = 'AD_classification'

elif task == 3:
    for img in os.listdir(img_path + 'EMCI/mwp1'):
        img_fpath = img_path + "EMCI/mwp1/" + img
        if os.path.isfile(img_fpath) and img.split(".")[-1] == "nii":
            sample_name.append('EMCI/mwp1/' + img)
            labels.append(1)
    for img in os.listdir(img_path + 'LMCI/mwp1'):
        img_fpath = img_path + "LMCI/mwp1/" + img
        if os.path.isfile(img_fpath) and img.split(".")[-1] == "nii":
            sample_name.append('LMCI/mwp1/' + img)
            labels.append(0)
    task_name = 'MCI_conversion'
elif task == 2:
    for img in os.listdir(img_path + 'CN/mwp1'):
        img_fpath = img_path + "CN/mwp1/" + img
        if os.path.isfile(img_fpath) and img.split(".")[-1] == "nii":
            sample_name.append('CN/mwp1/' + img)
            labels.append(0)
    for img in os.listdir(img_path + 'EMCI/mwp1'):
        img_fpath = img_path + "EMCI/mwp1/" + img
        if os.path.isfile(img_fpath) and img.split(".")[-1] == "nii":
            sample_name.append('EMCI/mwp1/' + img)
            labels.append(1)
    task_name = 'CN_EMCI'
elif task == 4:
    for img in os.listdir(img_path + 'CN/mwp1'):
        img_fpath = img_path + "CN/mwp1/" + img
        if os.path.isfile(img_fpath) and img.split(".")[-1] == "nii":
            sample_name.append('CN/mwp1/' + img)
            labels.append(1)
    for img in os.listdir(img_path + 'LMCI/mwp1'):
        img_fpath = img_path + "LMCI/mwp1/" + img
        if os.path.isfile(img_fpath) and img.split(".")[-1] == "nii":
            sample_name.append('LMCI/mwp1/' + img)
            labels.append(0)
    task_name = 'CN_LMCI'

sample_name = np.array(sample_name)
print(len(sample_name))
labels = np.array(labels)
permut = np.random.permutation(len(sample_name))
np.take(sample_name, permut, out=sample_name)  #
np.take(labels, permut, out=labels)


fold_num = 5
samples_train, samples_test, labels_train, labels_test = train_test_split(sample_name, labels, test_size=0.2,
                                                                          random_state=1, stratify=labels)

samples_train_1 = samples_train.flatten()
labels_train_1 = labels_train.flatten()

sample_train_list = []
sample_valid_list = []

skf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=1).split(samples_train_1, labels_train_1)
for train_list, valid_list in skf:
     sample_valid_list.append(valid_list)
     sample_train_list.append(train_list)




a = img_path.split("/")[0]
b = img_path.split("/")[1]
data_mat_dir = a+'/'+b+'/'+'data_split'+'/'+task_name+'/'
word_name = os.path.exists(data_mat_dir)
if not word_name:
    os.makedirs(data_mat_dir)


sio.savemat('G:/data_split/{}/Cuingnet_{}_ONLY_VALID.mat'.format(task_name, task_name),
            {"samples_train": samples_train, "samples_test": samples_test,
             "labels_train": labels_train, "labels_test": labels_test,
             # "sample_train_labels": np.array(sample_train_labels), "sample_valid_labels": np.array(
             # sample_valid_labels), "sample_train_name": np.array(sample_train_name), "sample_valid_name": np.array(
             # sample_valid_name),
              "sample_valid_list": sample_valid_list,
             "sample_train_list": sample_train_list})

