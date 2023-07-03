
import torch
import torch.optim as optim
import numpy as np
from Net.pruning import prun





from DataLoader.Data_Loader2 import data_flow, tst_data_flow
import scipy.io as sio
import h5py
from sklearn.metrics import roc_curve, auc
import os
import tqdm
import time



start = time.perf_counter()


print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0]

task = 1
epoch = 50
learning_rate = 0.001
batch_size = 32
patch_size = 21
patch_num = 90
kernel_num = [32, 64, 128, 128, batch_size]

img_path = "G:\Remote Data Synchronizing Folders\predMRIdata\predCuingnet/"
if task == 1:
    task_name = 'AD_classification'
elif task == 2:
    task_name = 'CN_EMCI'
elif task == 3:
    task_name = 'MCI_conversion'
elif task == 4:
    task_name = 'CN_LMCI'

data = sio.loadmat('G:/data_split/{}/Cuingnet_AD_classification_ONLY_VALID.mat'.format(task_name))  #读取.mat文件
sample_name = data['samples_train'].flatten()  #
labels = data['labels_train'].flatten()  #

sample_valid_list = np.squeeze(data['sample_valid_list'])
sample_train_list = np.squeeze(data['sample_train_list'])
res = np.zeros(shape=(5, 4))


results_test_dir = 'G:/select_patch'+'/'+task_name+'/patch-acc-results.xlsx'
word_name = os.path.exists(results_test_dir)
if not word_name:
    os.makedirs(results_test_dir)





best = np.inf
best_acc = 0
sen_best =0
spe_best = 0
roc_auc_best = 0
epochs = 0
# 5 fold
for i in range(90):
    best = np.inf
    best_acc = 0
    sen_best = 0
    spe_best = 0
    roc_auc_best = 0
    epochs = 0
    for e in range(50):

        valid_list = np.squeeze(sample_valid_list[0])
        train_list = np.squeeze(sample_train_list[0])

        labels_train = labels[train_list]
        labels_valid = labels[valid_list]
        samples_train = sample_name[train_list]
        samples_valid = sample_name[valid_list]

        template_cors = h5py.File('G:PearsonCorrCoef_test\PatchSelectResult/'
                                  'Cuingnet_size{}__num{}.mat'.format(patch_size, patch_num), 'r')
        template_cors = template_cors['patch_centers']

        # build model
        model = prun(patch_num=1, feature_depth=kernel_num, num_classes=2)


        model = model.to(device).cuda()

        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)


        pos_num = np.sum(labels_valid == 1)
        neg_num = np.sum(labels_valid == 0)



        train_loader = data_flow(img_path, samples_train, labels_train, template_cors,
                                     batch_size, patch_size, patch_num)
        model.train()
        l = []
        a = []
        train_loss = 0
        for i_batch in range(len(train_list) // batch_size):
            inputs_prun, outputs = next(train_loader)

            inputs_prun = torch.from_numpy(inputs_prun[i]).to(device).cuda()
            subject_outputs = torch.from_numpy(outputs).long().flatten().to(device).cuda()

            model.zero_grad()
            optimizer.zero_grad()

            subject_pred = model(inputs_prun)
            loss = criterion(subject_pred, subject_outputs)

            loss.backward()
            optimizer.step()

            l.append(loss.item())
            subject_pred = subject_pred.max(1)[1]

            acc = torch.sum(torch.eq(subject_pred, subject_outputs)).cpu().numpy()

            a.append(acc / batch_size)



        acc = 0
        model.eval()
        TP = 0
        TN = 0
        subject_probs = []
        l_val = []
        val_loss = 0

        for i_batch in range(len(samples_valid)):
            inputs, outputs = tst_data_flow(img_path, samples_valid[i_batch], labels_valid[i_batch],
                                                template_cors, patch_size, patch_num)

            inputs = torch.from_numpy(inputs[i+60]).to(device).cuda()
            subject_outputs = torch.from_numpy(outputs).long().flatten().to(device).cuda()
            subject_pred = model(inputs)
            loss = criterion(subject_pred, subject_outputs)  # new
            l_val.append(loss.item())
            subject_prob = subject_pred.cpu().detach().numpy()[:, 1][0]
            subject_probs.append(subject_prob)
            subject_pred = subject_pred.max(1)[1]


            if subject_outputs.cpu().numpy()[0] == 1 and subject_pred.cpu().numpy()[0] == 1:
                    TP += 1
            if subject_outputs.cpu().numpy()[0] == 0 and subject_pred.cpu().numpy()[0] == 0:
                    TN += 1
            acc += torch.sum(torch.eq(subject_pred, subject_outputs)).cpu().numpy()

        acc = acc / len(valid_list)
        sen = TP / pos_num
        spe = TN / neg_num
        fpr, tpr, thresholds = roc_curve(labels_valid, subject_probs)
        roc_auc = auc(fpr, tpr)


        if best_acc < acc:
            best_acc = acc
            sen_best = sen
            spe_best = spe
            roc_auc_best = roc_auc
            epochs = e



    print('[patch{} validation] cur_acc:{:.4f} epochs{} is best_acc:{:.4f} '
          'cur_loss:{:.4f}'.format(i, acc, epochs, best_acc , np.mean(l_val)))
    print('当前时间为：')
    print('当前时间为：'+ time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))



end = time.perf_counter()
print("final is in : %s mins " % (int(end-start)/60))

save_results(res, task_name, end-start)


