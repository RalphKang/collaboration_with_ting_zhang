#%%
import pandas as pd
from torch.utils.data import DataLoader
import torch
import argparse
import json
import numpy as np
from Mydataset_stack_zhang import dataset_zhang_bio, fix_random_seeds
from network_repo import MLP_forward_embed
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
fix_random_seeds(seed=1234)


#%%
parser = argparse.ArgumentParser()

parser.add_argument('--device', type=str, default='cpu', help='device')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--model_save_dir', type=str, default="result/mlp_2in_embed_128_sim.pth", help='model save dir')
parser.add_argument('--performance_dir', type=str, default="result/qingjie_model/mlp_2in_embed_128_sim.txt", help='performance dir')
parser.add_argument('--dataset_dir', type=str, default="zhang_data/hm_feedback_sim.npy", help='dataset dir')
args=parser.parse_args()

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%


if args.model_save_dir == "result/mlp_2in_embed_128_sim.pth":
    test_dataset=dataset_zhang_bio(args=args,data_ratio=0.2,train_mode=False)
    vali_dl=DataLoader(test_dataset,batch_size=1,shuffle=False)
    layers=[128,64,32,16,1]
    model=MLP_forward_embed(layers,embedding_dim=int(128/2),input_dim=6).to(args.device)
    model.load_state_dict(torch.load(args.model_save_dir))
data_amount=int(len(test_dataset.orig_input)*0.2)
data_index=test_dataset.shuffle_index[-data_amount:]

# model.eval()


# input_test=test_dataset.input.float()
# target_test=test_dataset.label.float()

# input_test=input_test.to(args.device)

# pred_label=model(input_test)
# pred_label=pred_label.cpu()
# error=torch.abs(target_test-pred_label)
# error_num=torch.sum(error>=0.5)
# accuracy=1-error_num.item()/len(error)
# print(accuracy)


error_sum=0
    # with torch.no_grad():
with torch.no_grad():
    for j, (input_eval, target_eval) in enumerate(vali_dl):
        input_eval = input_eval.squeeze(1).to(args.device)
        target_eval = target_eval.to(args.device)
        pred_eval = model(input_eval)
        abs=torch.abs(pred_eval-target_eval)
        error_number=torch.sum(abs>=0.5)
        error_sum+=error_number
        # loss_eval = loss(out_eval_hat, target_eval)
        # loss_eval=loss_eval+pde_loss
    accuracy=1-error_sum/j        
    print(accuracy)
