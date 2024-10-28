import numpy as np
import scipy.io as sio
data_dir="zhang_data/"
data=np.load(data_dir+"hm_feedback_sim.npy")
# save data to mat
sio.savemat(data_dir+"hm_feedback_sim.mat",{'data':data})
print("----")