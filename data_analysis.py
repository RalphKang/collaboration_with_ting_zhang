import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

data_dir="data_from_qingjie/"
background_data=loadmat(data_dir+"wave_background.mat")
background_data=background_data["P_snapshot1"]
background_data_real=background_data.real

print(background_data_real.shape)
plt.imshow(background_data_real)
plt.colorbar()
plt.savefig("background_data_real.png")
plt.close()

anormaly_data=loadmat(data_dir+"wave_anormaly.mat")
anormaly_data=anormaly_data["P_snapshot1"]
anormaly_data_real=anormaly_data.real

print(anormaly_data_real.shape)
plt.imshow(anormaly_data_real)
plt.colorbar()
plt.savefig("anormaly_data_real.png")
plt.close()
du=anormaly_data_real-background_data_real

plt.imshow(du)
plt.colorbar()
plt.savefig("du.png")
plt.close()

du_image=anormaly_data.imag-background_data.imag

plt.imshow(du_image)
plt.colorbar()
plt.savefig("du_image.png")
plt.close()


input_back_dir=data_dir+"v_background.txt"
input_anormaly_dir=data_dir+"v_anormaly.txt"

input_bkgrd=np.loadtxt(input_back_dir)
input_anormaly=np.loadtxt(input_anormaly_dir)

plt.imshow(input_bkgrd)
plt.colorbar()
plt.savefig("input_bkgrd.png")
plt.close()

plt.imshow(input_anormaly)
plt.colorbar()
plt.savefig("input_anormaly.png")
plt.close()
print("-----")
m0=1/((input_bkgrd/1000)**2)
m1=1/((input_anormaly/1000)**2)

plt.imshow(m0)
plt.colorbar()
plt.savefig("m0.png")
plt.close()

plt.imshow(m1)
plt.colorbar()
plt.savefig("m1.png")
plt.close()