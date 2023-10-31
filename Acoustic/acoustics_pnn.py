import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from torch.utils.data import DataLoader
from collections import defaultdict
import pandas as pd
from random import choice
import os
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import io
from google.colab import files
import matplotlib.pyplot as plt

# Vowel datasets

def load_Vowel():

  # read by default 1st sheet of an excel file
  dataframe1 = pd.read_excel('Vowel_dataset.xlsx')

  data=np.array(dataframe1)
  Data=data
  Data_class=[]
  for i in range(37):
        Data[i,0]=0
  for i in range(37,74,1):
        Data[i,0]=1
  for i in range(74,111,1):
        Data[i,0]=2
  for i in range(111,148,1):
        Data[i,0]=3
  for i in range(148,185,1):
        Data[i,0]=4
  for i in range(185,222,1):
        Data[i,0]=5
  for i in range(222,258,1):
        Data[i,0]=6
  dim = 0
  idx = torch.randperm(Data.shape[dim])

  Data_shuffled = Data[idx]
  Data_shuffled

  Train_data=[]
  Test_data=[]
  Train_label=[]
  Test_label=[]

  Full_data=[]
  Full_label=[]
  for i in range(Data_shuffled.shape[0]):
    Full_data.append(Data_shuffled[i,1:13])
    Full_label.append(Data_shuffled[i,0])

  Full_data=np.array(Full_data)
  Full_label=np.array(Full_label)

  for i in range(12):
    Full_data[:,i] = (Full_data[:,i] -Full_data[:,i].mean()) / Full_data[:,i].std()

  Full_data=Full_data-np.min(Full_data)
  for i in range(Full_data.shape[0]):
    if i<int(.8*Full_data.shape[0]):
      Train_data.append(Full_data[i,:])
      Train_label.append(Full_label[i])
    else:
      Test_data.append(Full_data[i,:])
      Test_label.append(Full_label[i])

  Train_data=np.array((Train_data), dtype=np.float32)
  Test_data=np.array((Test_data), dtype=np.float32)
  Train_label=np.array(Train_label)
  Test_label=np.array(Test_label)
  X=8
  Train_Data=[]
  for i in range(Train_data.shape[0]):
    Train_Data.append(np.concatenate(( np.zeros(X) ,Train_data[i,:]), axis=None))
  Train_Data=np.array((Train_Data), dtype=np.float32)

  Test_Data=[]
  for i in range(Test_data.shape[0]):
    Test_Data.append(np.concatenate(( np.zeros(X) ,Test_data[i,:]), axis=None))
  Test_Data=np.array((Test_Data), dtype=np.float32)


  train_batch_size = (Train_data).shape[0]
  val_batch_size = (Test_data).shape[0]


  train_data = []
  for i in range(Train_data.shape[0]):
    train_data.append([Train_Data[i], Train_label[i]])

  test_data = []
  for i in range(Test_data.shape[0]):
    test_data.append([Test_Data[i], Test_label[i]])

  train_loader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
  val_loader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=val_batch_size)
  return train_loader,val_loader




p_vector0=torch.normal(0, 1, size=(1, 20))
p_vector0=p_vector0/ (p_vector0.norm(2, 1, keepdim=True) + 1e-4)
p_vector1=torch.normal(0, 1, size=(1,20))
p_vector1=p_vector1/ (p_vector1.norm(2, 1, keepdim=True) + 1e-4)
p_vector0=p_vector0.repeat(270, 1)
p_vector0=p_vector0.cuda()
p_vector1=p_vector1.repeat(270, 1)
p_vector1=p_vector1.cuda()



def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :7] *= 0.0
    x_[range(x.shape[0]), y] = (torch.abs(x).max())
    return x_


Scale=1
N_F=1
acc_test=[]
acc_train=[]
class FNet(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        self.para_weight=[]
        self.para_bias=[]
        self.loss_save=defaultdict(list)
        self.param_save=[]
        self.num_epochs =200
        self.num_label=7
        self.N_layer=int(len(dims)/2)
        self.loss_list=[0] * int(len(dims)/2)
        for d in range(0, (len(dims)), 2):
            self.layers += [Layer( dims[d ], dims[d+1 ]).cuda()]

    def predict(self, x):
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        goodness_per_label = []
        for label in range(self.num_label):
            h = overlay_y_on_x(x, label)

            goodness = []
            for k, layer in enumerate(self.layers):
              h_t=torch.cat((self.PNN_Acoustic(h),h), 1)
              h = layer(h_t,k)
              g=torch.abs(h)
              if k==0:
                goodness += [cos(g, p_vector0[:h.shape[0]])]
              if k==1:
                goodness += [cos(g, p_vector1[:h.shape[0]])]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)


# Forward model of Acoustic system
    def PNN_Acoustic(self,x):
      model.eval()
      x=torch.abs(x)/(x.norm(2,1,keepdim=True)+1e-6)
      outputs = model(x)
      return outputs.detach()



    def train(self, train_loader,val_loader):
        k=0
        for i in range(self.num_epochs):
            self.loss_list=[0] * self.N_layer
            for j, data in enumerate(train_loader, 0):
              x, y = data
              x, y = x.cuda(),y.cuda()
              x_pos = overlay_y_on_x(x, y)
              y_n=y
              for s in range(x.size(0)):
                y_n[s]=int(choice(list(set(range(0, 7)) - set([y[s].item()]))))
              x_neg = overlay_y_on_x(x, y_n)
              h_pos, h_neg = x_pos.cuda(), x_neg.cuda()

              for k, layer in enumerate(self.layers):
                p_t=torch.cat((self.PNN_Acoustic(h_pos),h_pos), 1)
                n_t=torch.cat((self.PNN_Acoustic(h_neg),h_neg), 1)
                h_pos, h_neg,loss = layer.train(p_t,  n_t,k)
                self.loss_list[k]=self.loss_list[k]+loss
              self.loss_list=np.divide(self.loss_list, j+1)
              er=0.0
              for g, data in enumerate(train_loader, 0):
                x_t, y_t = data
                x_t, y_t = x_t.cuda(), y_t.cuda()
                er+=( net.predict(x_t).eq(y_t).float().mean().item())

              print('Train Accuracy:', (er)/(g+1)*100)
              acc_train.append((er)/(g+1)*100)
              er=0.0
              for t, data in enumerate(val_loader, 0):
                x_te, y_te = data
                x_te, y_te = x_te.cuda(), y_te.cuda()
                er+=( net.predict(x_te).eq(y_te).float().mean().item())
              print('Test Accuracy:', (er)/(t+1)*100)
              acc_test.append((er)/(t+1)*100)

            for l in range(self.N_layer):
                  self.loss_save[l].append(self.loss_list[l])
            print( f'[Epochs: {i + 1}], Loss: {self.loss_list}' )

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.P=nn.Linear(in_features,out_features*N_F,bias=False).cuda()
        self.P1=nn.Linear(in_features,out_features*N_F,bias=False).cuda()
        self.P0=nn.Linear(in_features,out_features,bias=True).cuda()
        self.opt = Adam(list(self.P0.parameters()),weight_decay=0, lr=0.0005)
        self.threshold = 4
        self.running_loss=0.0
        self.num_epochs_internal =10




    def forward(self, x,k):
        return (self.P0(x))

    def goodness(self,x_pos,x_neg,k):
      cos = nn.CosineSimilarity(dim=1, eps=1e-6)
      if k==0:
        g_p = torch.abs(self.forward(x_pos,k))
        g_pos =cos(g_p, p_vector0[:x_pos.shape[0]])

        g_n = torch.abs(self.forward(x_neg,k))
        g_neg =cos(g_n, p_vector0[:x_pos.shape[0]])
      if k==1:
        g_p = torch.abs(self.forward(x_pos,k))
        g_pos =cos(g_p, p_vector1[:x_pos.shape[0]])

        g_n = torch.abs(self.forward(x_neg,k))
        g_neg =cos(g_n, p_vector1[:x_pos.shape[0]])

      return g_pos,g_neg


    def train(self, x_pos, x_neg,k):
        self.running_loss=0.0
        for i in range(self.num_epochs_internal):

          g_pos,g_neg=self.goodness(x_pos,x_neg,k)


          delta=g_pos-g_neg
          loss = (torch.log(1 + torch.exp(
              -self.threshold*delta ))).mean()

          self.opt.zero_grad()
          loss.backward()
          self.opt.step()
          self.running_loss+=loss.item()

        return self.forward(x_pos,k).detach(), self.forward(x_neg,k).detach(), self.running_loss/self.num_epochs_internal

torch.set_default_dtype(torch.float32)
acc_test=[]
acc_train=[]


# Run for 10 different seeds
for u in range(10):
  train_loader,val_loader=load_Vowel()
  net = FNet([40,20,40,20])
  net.train(train_loader,val_loader)

N_epoch=200
k=0
acc_train_avg=np.zeros((1,N_epoch))
acc_test_avg=np.zeros((1,N_epoch))
for i in range(len(acc_train)):
  if i%N_epoch==N_epoch-1 :
    acc_train_avg[0:N_epoch]+=np.array(acc_train[N_epoch*(k):N_epoch*(k+1)])
    acc_test_avg[0:N_epoch]+=np.array(acc_test[N_epoch*(k):N_epoch*(k+1)])
    k=k+1
acc_train_avg=acc_train_avg/k
acc_test_avg=acc_test_avg/k
plt.plot(acc_train_avg[0,:])
plt.plot(acc_test_avg[0,:])
plt.show()


