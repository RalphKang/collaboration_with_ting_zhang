# %%
from sys import exit
import json

import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.optim as optim

from Mydataset_stack_zhang import dataset_zhang_bio,fix_random_seeds
import torch.nn as nn
from train_vali_function_v3 import train_function, vali_function, get_lr, warm_up_lr
import argparse
import scipy.io
import torch
import random
from network_repo import MLP_forward, MLP_forward_fourier, MLP_forward_embed

"""
This code is used to train supervised NN with Qingjie's data, all modules included herein are for supervised learning
"""
# %% overall configuration----------------------------------------------------------
# def main(configure_file_dir):
    #read the configuration file
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epoch', type=int, default=10000, help='epoch')
parser.add_argument('--warm_up_epoch', type=int, default=10, help='warm up epoch')
parser.add_argument('--search_lr', type=bool, default=False, help='search lr')
parser.add_argument('--pretrain_mode', type=bool, default=False, help='pretrain mode')
parser.add_argument('--model_save_dir', type=str, default="result/mlp_2in_embed_128_sim.pth", help='model save dir')
parser.add_argument('--performance_dir', type=str, default="result/mlp_2in_embed_128_sim.txt", help='performance dir')
parser.add_argument('--lr', type=float, default=0.0, help='learning rate,0.0 means do not use this learning rate')
parser.add_argument('--dataset_dir', type=str, default="zhang_data/hm_feedback_sim.npy", help='dataset dir')
parser.add_argument("--sensor_number", type=int, default=1, help="number of sensors")

args = parser.parse_args()

pretrain_mode = args.pretrain_mode
search_lr = args.search_lr
epoch0 = args.warm_up_epoch
epoch = args.epoch
guide_wd = 0.00
pretrain_mode = args.pretrain_mode
search_lr = args.search_lr
epoch0 = args.warm_up_epoch
epoch1=5000
epoch = args.epoch
initial_lr = 3.0e-2  # no need to change for pretrain


#%% load dataset---------------------------
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fix_random_seeds(seed=1234)



# set model
if args.model_save_dir == "result/qingjie_model/mlp_3in.pth":
    dataset_train=dataset_wavefield_without_u0(args=args,data_ratio=0.7,train_mode=True)
    train_loader=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)

    dataset_vali=dataset_wavefield_without_u0(args=args,data_ratio=0.3,train_mode=False)
    vali_loader=DataLoader(dataset_vali,batch_size=args.batch_size,shuffle=False)
    layers=[3,256,128,64,32,16,2]
    model=MLP_forward(layers).to(args.device)
elif args.model_save_dir == "result/qingjie_model/mlp_6in.pth":
    dataset_train=dataset_wavefield(args=args,data_ratio=0.7,train_mode=True)
    train_loader=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)

    dataset_vali=dataset_wavefield(args=args,data_ratio=0.3,train_mode=False)
    vali_loader=DataLoader(dataset_vali,batch_size=args.batch_size,shuffle=False)
    layers=[6,128,64,32,16,2]
    model=MLP_forward(layers).to(args.device)
elif args.model_save_dir == "result/qingjie_model/mlp_3in_fourier.pth":
    dataset_train=dataset_wavefield_without_u0(args=args,data_ratio=0.7,train_mode=True)
    train_loader=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)

    dataset_vali=dataset_wavefield_without_u0(args=args,data_ratio=0.3,train_mode=False)
    vali_loader=DataLoader(dataset_vali,batch_size=args.batch_size,shuffle=False)
    layers=[64,128,64,32,16,2]
    model=MLP_forward_fourier(layers,embedding_dim=64,input_dim=3).to(args.device)
elif args.model_save_dir == "result/qingjie_model/mlp_2in_fourier.pth":
    dataset_train=dataset_wavefield_xz(args=args,data_ratio=0.7,train_mode=True)
    train_loader=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)

    dataset_vali=dataset_wavefield_xz(args=args,data_ratio=0.3,train_mode=False)
    vali_loader=DataLoader(dataset_vali,batch_size=args.batch_size,shuffle=False)
    layers=[128,128,64,32,16,2]
    model=MLP_forward_fourier(layers,embedding_dim=128,input_dim=2).to(args.device)
elif args.model_save_dir == "result/qingjie_model/mlp_2in_fourier_64.pth":
    dataset_train=dataset_wavefield_xz(args=args,data_ratio=0.7,train_mode=True)
    train_loader=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)

    dataset_vali=dataset_wavefield_xz(args=args,data_ratio=0.3,train_mode=False)
    vali_loader=DataLoader(dataset_vali,batch_size=args.batch_size,shuffle=False)
    layers=[64,128,64,32,16,2]
    model=MLP_forward_fourier(layers,embedding_dim=64,input_dim=2).to(args.device)
elif args.model_save_dir == "result/mlp_2in_embed_128.pth":
    dataset_train=dataset_zhang_bio(args=args,data_ratio=0.8,train_mode=True)
    train_loader=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)

    dataset_vali=dataset_zhang_bio(args=args,data_ratio=0.2,train_mode=False)
    vali_loader=DataLoader(dataset_vali,batch_size=args.batch_size,shuffle=False)
    layers=[128,64,32,16,1]
    model=MLP_forward_embed(layers,embedding_dim=int(128/2),input_dim=6).to(args.device)
elif args.model_save_dir == "result/mlp_2in_embed_128_sim.pth":
    dataset_train=dataset_zhang_bio(args=args,data_ratio=0.8,train_mode=True)
    train_loader=DataLoader(dataset_train,batch_size=args.batch_size,shuffle=True)

    dataset_vali=dataset_zhang_bio(args=args,data_ratio=0.2,train_mode=False)
    vali_loader=DataLoader(dataset_vali,batch_size=args.batch_size,shuffle=False)
    layers=[128,64,32,16,1]
    model=MLP_forward_embed(layers,embedding_dim=int(128/2),input_dim=6).to(args.device)

# layers=[3,256,128,64,32,16,2] # PINN_V1
input_upper=dataset_train.input_max
input_upper=input_upper.to(args.device)
input_lower=dataset_train.input_min
input_lower=input_lower.to(args.device)



# setting of loss,optimizer,lr_schedule and early stop----------------------------------
model_save_dir = args.model_save_dir
performance_dir = args.performance_dir

if pretrain_mode:
    model.load_state_dict(torch.load(model_save_dir))
    history = np.loadtxt(performance_dir)

    historical_best = np.max(history[:, 3])
    initial_lr = history[np.where(history[:, 3] == historical_best), 0]
    # print("historical best obtained is {:.6f}".format(historical_best))
else:
    historical_best = 0.0
if args.lr > 0.0:
    initial_lr = args.lr
if search_lr:
    initial_lr = 1e-8
    pretrain_mode = False

optimizer = Adam(model.parameters(), lr=1.5e-3, weight_decay=guide_wd)
optimizer_lbfgs=optim.LBFGS(model.parameters(), lr=initial_lr, max_iter=10, max_eval=10, history_size=10, line_search_fn="strong_wolfe")
lr_scd = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
# lr_scd= ReduceLROnPlateau(optimizer, 'min',cooldown=3)
loss = nn.BCELoss(reduction="mean")

train_loss_record = []
vali_loss_record = []
lr_record = []

# -------------warm up training-----------------------------------------
if not pretrain_mode:
    for ep in range(epoch0):
        warm_up_lr(initial_lr=initial_lr, optimizer=optimizer, ite=ep, boundary=epoch0)

        train_loss_record = train_function(device=args.device, model=model, train_dl=train_loader, optimizer=optimizer,
                                            loss=loss, ep=ep,
                                            epoch=epoch0,
                                            train_loss_record=train_loss_record, lr_search=search_lr)
        if search_lr:
            exit()
        vali_loss_record, historical_best,average_acurracy = vali_function(device=args.device, model=model, model_save_dir=model_save_dir,
                                                            vali_dl=vali_loader,
                                                            loss=loss, ep=ep,
                                                            epoch=epoch0, vali_loss_record=vali_loss_record,
                                                            historical_best=historical_best)

        # vital parameters record
        old_lr = get_lr(optimizer)
        lr_record.append(old_lr)
        f = open(performance_dir, 'a')  # open file in append mode
        np.savetxt(f, np.c_[
            old_lr, train_loss_record[-1].cpu().detach().numpy(), vali_loss_record[-1].cpu().detach().numpy()])
        f.close()
#--------------------official training_1st Stage--------------------------------
for ep in range(epoch1):
    # change_lr(initial_lr,optimizer=optimizer,ite=ep,mode="exp",scale=30)
    print("official training")
    train_loss_record = train_function(device=args.device, model=model, train_dl=train_loader, optimizer=optimizer,
                                            loss=loss, ep=ep,
                                            epoch=epoch1,
                                            train_loss_record=train_loss_record, lr_search=False)
    vali_loss_record, historical_best,average_acurracy = vali_function(device=args.device, model=model, model_save_dir=model_save_dir,
                                                        vali_dl=vali_loader,
                                                        loss=loss, ep=ep, epoch=epoch,
                                                        vali_loss_record=vali_loss_record,
                                                        historical_best=historical_best)
    # learning rate on plateau change--
    old_lr = get_lr(optimizer)
    lr_scd.step()
    lr_record.append(old_lr)
    f = open(performance_dir, 'a')  # open file in append mode
    np.savetxt(f, np.c_[
        old_lr, train_loss_record[-1].cpu().detach().numpy(), vali_loss_record[-1].cpu().detach().numpy(),average_acurracy.cpu().detach().numpy()])
    f.close()

# if __name__ == '__main__':
#     configure_file_dir = './configure_file/resnet_single_round.json'  # change this to your configure file
#     main(args=args, configure_file_dir=configure_file_dir)
