import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
import torch.utils.data as data_utils
from torch.autograd import Variable
import pickle
from sklearn.model_selection import train_test_split
import math          
import re
import collections
from functools import partial
from torch import optim
from torchsummary import summary
from model import DeepNeo
from Radam import RAdam
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
import time
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import sys


def precision_recall(y_true, y_pred):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''

    tp = (y_true * y_pred).sum().to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
    
    epsilon = 1e-7
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    
    f1 = 2 * (precision*recall) / (precision + recall + epsilon)

    return precision, recall, f1


def train_model_5cv(num_epochs=300):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kfold = KFold(n_splits=5, shuffle=True)
    
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        globals()[f'{fold}_result'] = []
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        model = DeepNeo.from_name(f'DeepNeo-{allele}-{length}-short')
        criterion = nn.BCELoss()
        optimizer = RAdam(model.parameters())
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
        
        model.to(device)
        criterion.to(device)
        
        best_model_wts = model.state_dict()
        best_loss = 1000.0
        
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
        # Define data loaders for training and testing data in this fold
        trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=10, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=10, sampler=test_subsampler)
        
        dataloaders = {'train': trainloader, 'valid': testloader}
        dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'valid']}
        
        for epoch in tqdm(range(num_epochs), position=0, leave=True):
            print('-' * 60)
            print('Epoch {}/{}'.format(epoch+1, num_epochs))


            for mode in dataloaders:
                loss_ = 0.0
                corrects_ = 0.0
                precision_,recall_, f1_ = 0.0, 0.0, 0.0

                if mode == 'train':
                    # training step
                    model.train(True)
                else:
                    # Validation step
                    model.train(False)

                for data in tqdm(dataloaders[mode],  position=1, leave=True):
                     # get the inputs
                    inputs, labels = data
                    inputs = Variable(inputs.to(device, dtype=torch.float), requires_grad=True)
                    labels = Variable(labels.to(device))

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    outputs = model(inputs)
                    precision, recall, f1 = precision_recall(labels.float().view(-1,1), outputs)
                    loss = criterion(outputs, labels.float().view(-1,1)).to(device)
                    if mode == 'train':
                    # backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                    # statistics
                    loss_ += loss.data
                    precision_ += precision.data
                    recall_ += recall.data
                    f1_ += f1.data
                    preds = (outputs>=0.5).float()
                    corrects_ += accuracy_score(labels.cpu(), preds.cpu())

                if mode == 'train': 
                    epoch_train_loss = loss_ / dataset_sizes[f'train']
                    epoch_train_precision = precision_ / dataset_sizes[f'train']
                    epoch_train_recall = recall_ / dataset_sizes[f'train']
                    epoch_train_f1 = f1_ / dataset_sizes[f'train']
                    epoch_train_acc = corrects_ / dataset_sizes[f'train']
                    print(f'train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} F1: {epoch_train_f1:.4f} Precision: {epoch_train_precision:.4f} Recall: {epoch_train_recall:.4f}')
                else:
                    epoch_val_loss = loss_ / dataset_sizes[f'valid']
                    epoch_val_precision = precision_ / dataset_sizes[f'valid']
                    epoch_val_recall = recall_ / dataset_sizes[f'valid']
                    epoch_val_f1 = f1_ / dataset_sizes[f'valid']
                    epoch_val_acc = corrects_ / dataset_sizes[f'valid']
                    globals()[f'{fold}_result'].append(epoch_val_precision)
                    print(f'valid Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f} F1: {epoch_val_f1:.4f} Precision: {epoch_val_precision:.4f} Recall: {epoch_val_recall:.4f}')


            # epoch마다 아래 정보를 출력
            writer.add_scalars('Loss', {f'train_{fold}':epoch_train_loss, f'validation_{fold}':epoch_val_loss}, epoch)
            writer.add_scalars('Accuracy', {f'train_{fold}':epoch_train_acc, f'validation_{fold}':epoch_val_acc},  epoch)
            writer.add_scalars('F1', {f'train_{fold}':epoch_train_f1, f'validation_{fold}':epoch_val_f1},  epoch)
            writer.add_scalars('precision', {f'train_{fold}':epoch_train_precision, f'validation_{fold}':epoch_val_precision},  epoch)
            writer.add_scalars('recall', {f'train_{fold}':epoch_train_recall, f'validation_{fold}':epoch_val_recall},  epoch)

            # deep copy the model
            if  epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                best_model_wts = model.state_dict()
                # save
                checkpoint = {'model': model,
                          #'state_dict': model.module.state_dict(),
                              'state_dict': model.state_dict(),
                          'optimizer' : optimizer.state_dict()}

                savePath = "{}/{}fold_best_{}.pth".format(save_dir, fold, epoch+1)
                torch.save(checkpoint, savePath)
            else:
                # save
                checkpoint = {'model': model,
                          #'state_dict': model.module.state_dict(),
                              'state_dict': model.state_dict(),
                          'optimizer' : optimizer.state_dict()}

                savePath = "{}/{}fold_{}.pth".format(save_dir, fold,epoch+1)
                torch.save(checkpoint, savePath)


            print('-' * 60)
            print()
            
        torch.cuda.empty_cache() 

    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_best_model(num_epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    since = time.time()
    model = DeepNeo.from_name(f'DeepNeo-{allele}-{length}-short')
    criterion = nn.BCELoss()
    optimizer = RAdam(model.parameters())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    model.to(device)
    criterion.to(device)

    best_model_wts = model.state_dict()
    best_loss = 1000.0

    for epoch in tqdm(range(num_epochs), position=0, leave=True):
        print('-' * 60)
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        loss_ = 0.0
        corrects_ = 0.0
        precision_,recall_, f1_ = 0.0, 0.0, 0.0

        model.train(True)
    
        for data in tqdm(dataloaders):
             # get the inputs
            inputs, labels = data
            inputs = Variable(inputs.to(device, dtype=torch.float), requires_grad=True)
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            outputs = model(inputs)
            precision, recall, f1 = precision_recall(labels.float().view(-1,1), outputs)
            loss = criterion(outputs, labels.float().view(-1,1)).to(device)

            # backward + optimize only if in training phase

            loss.backward()
            optimizer.step()

            # statistics
            loss_ += loss.data
            precision_ += precision.data
            recall_ += recall.data
            f1_ += f1.data
            preds = (outputs>=0.5).float()
            corrects_ += accuracy_score(labels.cpu(), preds.cpu())

        epoch_train_loss = loss_ / dataset_sizes
        epoch_train_precision = precision_ / dataset_sizes
        epoch_train_recall = recall_ / dataset_sizes
        epoch_train_f1 = f1_ / dataset_sizes
        epoch_train_acc = corrects_ / dataset_sizes
        print(f'train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} F1: {epoch_train_f1:.4f} Precision: {epoch_train_precision:.4f} Recall: {epoch_train_recall:.4f}')


        # epoch마다 아래 정보를 출력
        writer.add_scalars('Loss' , {'train':epoch_train_loss}, epoch)
        writer.add_scalars('Accuracy' , {'train':epoch_train_acc},  epoch)
        writer.add_scalars('F1' , {'train':epoch_train_f1},  epoch)
        writer.add_scalars('precision' , {'train':epoch_train_precision},  epoch)
        writer.add_scalars('recall' , {'train':epoch_train_recall},  epoch)


        best_model_wts = model.state_dict()
        # save
        checkpoint = {'model': model,
                  #'state_dict': model.module.state_dict(),
                      'state_dict': model.state_dict(),
                  'optimizer' : optimizer.state_dict()}

        savePath = "{}/best_{}.pth".format(save_dir, epoch+1)
        torch.save(checkpoint, savePath)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":
    allele = sys.argv[1]
    length = sys.argv[2]
    dataset = sys.argv[3]
    try:
        gpu_num = sys.argv[4]
    except:
        gpu_num = 0

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num
    print(f'Using CUDA_VISIBLE_DEVICES {gpu_num}')
    model = DeepNeo.from_name(f'DeepNeo-{allele}-{length}-short')

    if dataset == 'random':
        df = pd.read_pickle('MS_BA_training_set.pkl')
    else:
        df = pd.read_pickle('MS_BA_natural_protein_false_training_set.pkl')
    df = df[df['allele'].str.contains(allele)]
    df = df[df['length'] == int(length)]

    tmp = []
    for line in df.to_numpy():
        tmp.append(line[0] + '-' + str(line[3]))

    df['stratify'] = tmp
    df_ = df[df['stratify'].isin(df['stratify'].value_counts()[df['stratify'].value_counts()==1].index.tolist())]
    df = df[~df['stratify'].isin(df['stratify'].value_counts()[df['stratify'].value_counts()==1].index.tolist())]

    matrix = []
    for i in df['matrix'].to_numpy():
        matrix.append(i)
        
    matrix = np.array(matrix)
    matrix.shape
    answer = list(df['answer'].astype('int'))

    matrix_ = []
    for i in df_['matrix'].to_numpy():
        matrix_.append(i)
        
    matrix_ = np.array(matrix_)
    matrix_.shape
    answer_ = list(df_['answer'].astype('int'))

    xTrain, xTest, yTrain, yTest = train_test_split(matrix, 
                                                    list(answer), 
                                                    test_size=0.15,
                                                    random_state=42,
                                                    stratify=df['stratify'])

    try:
        xTrain = np.concatenate([xTrain, matrix_])
        yTrain = np.concatenate([yTrain, answer_])
    except:
        pass

    counts = np.bincount(yTrain)
    print('Number of positive samples in training data: {} ({:.2f}% of total)'.format(counts[1], 100 * float(counts[1]) / len(yTrain)))
    counts = np.bincount(yTest)
    print('Number of positive samples in validation data: {} ({:.2f}% of total)'.format(counts[1], 100 * float(counts[1]) / len(yTest)))

    BATCH_SIZE = 256
    train_set = data_utils.TensorDataset(torch.tensor(xTrain), torch.tensor(yTrain))
    valid_set = data_utils.TensorDataset(torch.tensor(xTest), torch.tensor(yTest))

    train_loader = data_utils.DataLoader(train_set, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
    valid_loader = data_utils.DataLoader(valid_set, batch_size=BATCH_SIZE, )

    dataloaders = {'train' : train_loader, 'valid' : valid_loader}
    dataset_sizes = {x: len(dataloaders[x]) for x in ['train', 'valid']}
    dataset = ConcatDataset([train_set, valid_set]) 
    
    # Writer will output to ./runs/ directory by default
    if dataset == 'random':
        save_dir = f'saved_model/DeepNeo_Sep_18_{allele}_{length}'
    else:
        save_dir = f'saved_model/DeepNeo_Sep_18_natural_protein_{allele}_{length}'
    writer = SummaryWriter(save_dir)
    
    model = train_model_5cv(500)

    for fold in range(5):
        tmp = []
    for i in globals()[f'{fold}_result']:
        tmp.append(float(i.cpu()))
    globals()[f'{fold}_result'] = tmp

    best_epoch = 0
    best_score = 0
    for i in range(500):
        tmp = 0
        for fold in range(5):
            tmp+=globals()[f'{fold}_result'][i]
        tmp = tmp/5
        
        if tmp > best_score:
            best_score = tmp
            best_epoch = i

    dataset = ConcatDataset([train_set, valid_set])
    dataloaders = data_utils.DataLoader(dataset, batch_size=BATCH_SIZE, pin_memory=True, shuffle=True)
    dataset_sizes = len(dataloaders)
    if dataset == 'random':
        save_dir = f'saved_model/DeepNeo_MHC_random_protein_{allele}_{length}_final'
    else:
        save_dir = f'saved_model/DeepNeo_MHC_natural_protein_{allele}_{length}_final'
    writer = SummaryWriter(save_dir)
    model = train_best_model(best_epoch)
