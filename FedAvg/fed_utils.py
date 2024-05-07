
import os
import logging
import os
from pickle import FALSE
import pandas as pd
from torchvision.io import read_image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import random
import numpy as np
from random import randrange, seed

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import sklearn.preprocessing as preprocessing

logger = logging.getLogger(__name__)



        # Swap the items from the two chosen indices
#######################
# TensorBaord setting #
#######################
def launch_tensor_board(log_path, port, host):
    """Function for initiating TensorBoard.
    
    Args:
        log_path: Path where the log is stored.
        port: Port number used for launching TensorBoard.
        host: Address used for launching TensorBoard.
    """
    os.system(f"tensorboard --logdir={log_path} --port={port} --host={host}")
    return True

#########################
# Weight initialization #
#########################
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal).
        init_gain: Scaling factor for (normal | xavier | orthogonal).
    
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] ...initialization method [{init_type}] is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)   
    model.apply(init_func)

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights.
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.
    """
    if False: #if len(gpu_ids) > 0:
        
        #assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

#################
# Dataset split #
#################
# class CustomAdultDataset(Dataset):
#     def __init__(self, pdfile):
#             pan_file=pd.read_csv(pdfile)
#             x_data = pan_file.drop('salary',axis=1)
#             y_labels = pan_file['salary']
#             x = x_data.to_numpy()
#             y = y_labels.to_numpy()
#             self.data=torch.tensor(x,dtype=torch.float32)
#             self.targets=torch.tensor(y)

#     def __len__(self):
#         return len(self.targets)

#     def __getitem__(self, idx):
        
#         return self.data[idx], self.targets[idx]

class AdultDataset(Dataset):
    def __init__(self):


            features = ["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",  "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss", "Hours per week", "Country", "Target"] 
            train_url = '/dataset/adult.data' 
            test_url = '/dataset/adult.test' 

            # This will download 3.8M
            original_train = pd.read_csv(train_url, names=features, sep=r'\s*,\s*', engine='python', na_values="?")
            original_test = pd.read_csv(test_url, names=features, sep=r'\s*,\s*',engine='python', na_values="?", skiprows=1)
            num_train = len(original_train)
            original = pd.concat([original_train, original_test])
            roc_original = original
            labels = original['Target']
            labels = labels.replace('<=50K', 0).replace('>50K', 1)
            labels = labels.replace('<=50K.', 0).replace('>50K.', 1)
            
            del original["Education"]
            del original["Target"]
            binary_data = pd.get_dummies(original)
            sen_attr= binary_data['Sex_Male']
            feature_cols = binary_data[binary_data.columns[:-2]]
            scaler = preprocessing.StandardScaler()
            data = pd.DataFrame(scaler.fit_transform(feature_cols), columns=feature_cols.columns)
            data= data.to_numpy()
            labels= labels.to_numpy()
            self.sen_attr=sen_attr.to_numpy()
            self.data=torch.from_numpy(data.astype(np.float32))
            targets=torch.from_numpy(labels.astype(np.float32))
            self.sen_attr=torch.from_numpy(self.sen_attr.astype(np.float32))
            self.targets= targets.view(targets.shape[0],1)
            self.sen_attr= self.sen_attr.view(self.sen_attr.shape[0],1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        
        return self.data[idx], self.targets[idx]
    
    def getSenAttr(self,idx):
        return self.sen_attr[idx]


class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        s = self.tensors[2][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y, s

    def __len__(self):
        return self.tensors[0].size(0)

def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    """Split the whole dataset in IID or non-IID manner for distributing to clients."""
    print('num_clients',num_clients)
    dataset_name = dataset_name.upper()
    # get dataset from torchvision.datasets if exists
    if hasattr(torchvision.datasets, dataset_name):
        # set transformation differently per dataset
        if dataset_name in ["CIFAR10"]:
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
        elif dataset_name in ["MNIST"]:
            transform = torchvision.transforms.ToTensor()
        
        # prepare raw training & test datasets
        training_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = torchvision.datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    else:
        # dataset not found exception
        training_dataset=AdultDataset()
        test_dataset=AdultDataset()
        transform=False
        # error_message = f"...dataset \"{dataset_name}\" is not supported or cannot be found in TorchVision Datasets!"
        # raise AttributeError(error_message)

    # unsqueeze channel dimension for grayscale image datasets
    if training_dataset.data.ndim == 3: # convert to NxHxW -> NxHxWx1
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]
    
    # split dataset according to iid flag
    IID_SPLIT = True 
    if IID_SPLIT:
        # shuffle data
        shuffled_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffled_indices]
        sensitive_labels= training_dataset.sen_attr[shuffled_indices]

        # partition data into num_clients
        split_size = len(training_dataset) // num_clients


        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size),
                torch.split(torch.Tensor(sensitive_labels), split_size)
            )
        )
        # ratio = 0.5
        # split_datasets = list(
        #     zip(
        #         torch.split(torch.Tensor(training_inputs), [round(ratio*len(training_dataset)),round((1-ratio)*len(training_dataset))]),
        #         torch.split(torch.Tensor(training_labels), [round(ratio*len(training_dataset)),round((1-ratio)*len(training_dataset))]),
        #         torch.split(torch.Tensor(sensitive_labels), [round(ratio*len(training_dataset)),round((1-ratio)*len(training_dataset))])
        #     )
        # )

        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
            ]
        
     
    if False:

        n0 =len(training_dataset.data)//2
        n1 =len(training_dataset.data)//2

        print(type(training_dataset.targets))

        D_0=training_dataset.data[torch.where(training_dataset.targets==0)[0]]
        D_1=training_dataset.data[torch.where(training_dataset.targets==1)[0]]

        shuf0 = torch.randperm(len(D_0))
        shuf1 = torch.randperm(len(D_1))

        D_0=D_0[shuf0]
        D_1=D_1[shuf1]

        T_0=training_dataset.targets[torch.where(training_dataset.targets==0)[0]][shuf0]
        T_1=training_dataset.targets[torch.where(training_dataset.targets==1)[0]][shuf1]

        S_0=training_dataset.sen_attr[torch.where(training_dataset.targets==0)[0]][shuf0]
        S_1=training_dataset.sen_attr[torch.where(training_dataset.targets==1)[0]][shuf1]

        pr_T0=len(T_0)/len(training_dataset.data)
        pr_T1=len(T_1)/len(training_dataset.data)

        p00=0.95
        p10=1-p00

        p01=2*pr_T0 - p00
        p11=1-p01

        assert 0 < p00 < 1 and 0 < p10 < 1 and 0 < p01 < 1 and 0 < p11 < 1

        print('Do',{D_0.shape})
        mnmnm=int(n0*p00)
        print('n*p',{mnmnm})
        C_0=torch.cat((D_0[0:int(n0*p00),:],D_1[0:int(n0*p10),:]))
        print('shape',{D_0[0:int(n0*p00),:].shape})
        CT_0=torch.cat((T_0[0:int(n0*p00),:],T_1[0:int(n0*p10),:]))
        CS_0=torch.cat((S_0[0:int(n0*p00),:],S_1[0:int(n0*p10),:]))
        shufC0 = torch.randperm(len(C_0))
        C_0=C_0[shufC0]
        CT_0=CT_0[shufC0]
        CS_0=CS_0[shufC0]


        C_1=torch.cat((D_0[int(n0*p00):(int(n0*p00))+int((n1*p01)),:],D_1[int(n0*p10):int(n0*p10)+(int(n1*p11)),:]))
        CT_1=torch.cat((T_0[int(n0*p00):(int(n0*p00))+int((n1*p01)),:],T_1[int(n0*p10):int(n0*p10)+(int(n1*p11)),:]))
        CS_1=torch.cat((S_0[int(n0*p00):(int(n0*p00))+int((n1*p01)),:],S_1[int(n0*p10):int(n0*p10)+(int(n1*p11)),:]))

        shufC1 = torch.randperm(len(C_1))
        C_1=C_1[shufC1]
        CT_1=CT_1[shufC1]
        CS_1=CS_1[shufC1]


        C=torch.cat((C_0,C_1))
        C_T=torch.cat((CT_0,CT_1))
        C_S=torch.cat((CS_0,CS_1))

        split_size = len(C) // 2
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(C), split_size),
                torch.split(torch.Tensor(C_T), split_size),
                torch.split(torch.Tensor(C_S), split_size)
            )
        )

        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
            ]

    if False: 

        ratio= 0.5 #THIS IS RATIO

        n0 =round(len(training_dataset.data)*(ratio))
        n1 =round(len(training_dataset.data)*(1-ratio))

        

        D_0=training_dataset.data[torch.where(training_dataset.sen_attr==0)[0]]
        D_1=training_dataset.data[torch.where(training_dataset.sen_attr==1)[0]]

        shuf0 = torch.randperm(len(D_0))
        shuf1 = torch.randperm(len(D_1))

        D_0=D_0[shuf0]
        D_1=D_1[shuf1]

        T_0=training_dataset.targets[torch.where(training_dataset.sen_attr==0)[0]][shuf0]
        T_1=training_dataset.targets[torch.where(training_dataset.sen_attr==1)[0]][shuf1]

        S_0=training_dataset.sen_attr[torch.where(training_dataset.sen_attr==0)[0]][shuf0]
        S_1=training_dataset.sen_attr[torch.where(training_dataset.sen_attr==1)[0]][shuf1]

        pr_S0=len(S_0)/len(training_dataset.data)
        pr_S1=len(S_1)/len(training_dataset.data)

        p00=0.2
        p10=1-p00

        #p01=2*pr_S0 - p00

        p01=(pr_S0 - (p00*ratio))/(1-ratio)


        p11=1-p01
        print('p00,p10,p01,p11',{p00},{p10},{p01},{p11})
        assert 0 < p00 < 1 and 0 < p10 < 1 and 0 < p01 < 1 and 0 < p11 < 1
        
        C_0=torch.cat((D_0[0:int(n0*p00),:],D_1[0:int(n0*p10),:]))
        CT_0=torch.cat((T_0[0:int(n0*p00),:],T_1[0:int(n0*p10),:]))
        CS_0=torch.cat((S_0[0:int(n0*p00),:],S_1[0:int(n0*p10),:]))
        shufC0 = torch.randperm(len(C_0))
        C_0=C_0[shufC0]
        CT_0=CT_0[shufC0]
        CS_0=CS_0[shufC0]


        C_1=torch.cat((D_0[int(n0*p00):(int(n0*p00))+int((n1*p01)),:],D_1[int(n0*p10):int(n0*p10)+(int(n1*p11)),:]))
        CT_1=torch.cat((T_0[int(n0*p00):(int(n0*p00))+int((n1*p01)),:],T_1[int(n0*p10):int(n0*p10)+(int(n1*p11)),:]))
        CS_1=torch.cat((S_0[int(n0*p00):(int(n0*p00))+int((n1*p01)),:],S_1[int(n0*p10):int(n0*p10)+(int(n1*p11)),:]))

        shufC1 = torch.randperm(len(C_1))
        C_1=C_1[shufC1]
        CT_1=CT_1[shufC1]
        CS_1=CS_1[shufC1]


        C=torch.cat((C_0,C_1))
        C_T=torch.cat((CT_0,CT_1))
        C_S=torch.cat((CS_0,CS_1))


     

       # partition data into num_clients
        split_size = len(C) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(C), split_size),
                torch.split(torch.Tensor(C_T), split_size),
                torch.split(torch.Tensor(C_S), split_size)
            )
        )



        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
            ]


    if False: 

        ratio= 0.5

        D_11=training_dataset.data[torch.where((training_dataset.sen_attr==1 )&(training_dataset.targets==1))[0]]
        D_00=training_dataset.data[torch.where((training_dataset.sen_attr==0 )&(training_dataset.targets==0))[0]]
        D_10=training_dataset.data[torch.where((training_dataset.sen_attr==1 )&(training_dataset.targets==0))[0]]
        D_01=training_dataset.data[torch.where((training_dataset.sen_attr==0 )&(training_dataset.targets==1))[0]]

        print('shape',{D_11.shape},{D_00.shape},{D_10.shape},{D_01.shape})


        T_11=training_dataset.targets[torch.where((training_dataset.sen_attr==1 )&( training_dataset.targets==1 ))[0]]
        T_00=training_dataset.targets[torch.where((training_dataset.sen_attr==0 )&( training_dataset.targets==0 ))[0]]
        T_10=training_dataset.targets[torch.where((training_dataset.sen_attr==1 )&( training_dataset.targets==0 ))[0]]
        T_01=training_dataset.targets[torch.where((training_dataset.sen_attr==0 )&( training_dataset.targets==1 ))[0]]


        print('shape1',{T_11.shape},{T_00.shape},{T_10.shape},{T_01.shape})

        S_11=training_dataset.sen_attr[torch.where((training_dataset.sen_attr==1 )&( training_dataset.targets==1 ))[0]]
        S_00=training_dataset.sen_attr[torch.where((training_dataset.sen_attr==0 )&( training_dataset.targets==0 ))[0]]
        S_10=training_dataset.sen_attr[torch.where((training_dataset.sen_attr==1 )&( training_dataset.targets==0 ))[0]]
        S_01=training_dataset.sen_attr[torch.where((training_dataset.sen_attr==0 )&( training_dataset.targets==1 ))[0]]

        print('shape2',{S_11.shape},{S_00.shape},{S_10.shape},{S_01.shape})

        C=torch.cat((D_11,D_00,D_10,D_01))                                  
        C_T=torch.cat((T_11,T_00,T_10,T_01))
        C_S=torch.cat((S_11,S_00,S_10,S_01))

        
        lamda=0.9
      
        weight = 2-(lamda*2)
        count = len(C)
        n = int( count * weight ) # Set the number of iterations
        for ix in range(n):
            ix0 = randrange( count )
            ix1 = randrange( count )
            C [ ix0 ], C [ ix1 ] = C [ ix1 ], C [ ix0 ]
            C_T[ ix0 ], C_T[ ix1 ] = C_T[ ix1 ], C_T[ ix0 ]
            C_S[ ix0 ], C_S[ ix1 ] = C_S[ ix1 ], C_S[ ix0 ]


       # partition data into num_clients
        split_size = len(C) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(C), split_size),
                torch.split(torch.Tensor(C_T), split_size),
                torch.split(torch.Tensor(C_S), split_size)
            )
        )

        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
            ]


    return local_datasets, test_dataset
