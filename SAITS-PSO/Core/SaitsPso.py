import time

import numpy
import numpy as np
from pyswarm import pso
import matplotlib.pyplot as plt
from pypots.optim import AdamW
from pypots.optim.lr_scheduler import StepLR
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mse
import datetime
import os
from Core import *
import joblib
import json
import sys

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
sys.setrecursionlimit(10000)
current_dir = os.path.dirname(__file__)      # Core/
parent_dir = os.path.dirname(current_dir)    # 项目根目录/
dir = os.path.join(parent_dir, 'X')  # 项目根目录/X/
train_set,val_set = get_datasets_path(dir)
train_x = train_set['X']
eval_x = val_set['X']
eval_x_ori = val_set['X_ori']

train_x_reshape = numpy.reshape(train_x,[train_x.shape[0],50,40],order='F')
eval_x_reshape = numpy.reshape(eval_x,[eval_x.shape[0],50,40],order='F')
eval_x_ori_reshape = numpy.reshape(eval_x_ori,[eval_x_ori.shape[0],50,40],order='F')
train_set = {'X':train_x_reshape}
val_set = {'X':eval_x_reshape,'X_ori':eval_x_ori_reshape}
#
# Define the discrete range of parameters.
d_ffn_choices = [512, 1024]
fitness_history = []
# Define fitness function
def fitness_function(params):

    n_layers = round(params[0])
    d_ffn = d_ffn_choices[round(params[1])]
    lr = params[0]

    optimizer = AdamW(lr=lr, weight_decay=0.01)
    # initialization
    saits = SAITS(n_steps=50, n_features=40, n_layers=n_layers, d_model=256, n_heads=4,
                  d_k=64, d_v=64, d_ffn=d_ffn, dropout=0.1,
                  epochs=5, optimizer=optimizer, batch_size=1024,
                  model_saving_strategy=None, device="cuda:0")

    train_loss, val_loss = saits.fit(train_set, val_set)

    fitness = np.abs(0.1 * train_loss + 0.9 * val_loss)

    fitness_history.append((fitness, {
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "lr": lr,
    }))
    print("当前参数:", fitness_history[-1])

    return fitness  # PSO 寻找的是最小值


# Define the parameter range of PSO: use indexes to represent the values of discrete parameters.
lb = [1.5 + 1e-6,0,0.0001]  # [n_layers_index, d_model_heads_index, d_ffn_index, lr]
ub = [3.5 + 1e-6,1,0.00099]
best_params, best_score = pso(fitness_function, lb, ub, swarmsize=2, maxiter=10)# 执行 PSO 优化
n_layers = round(best_params[0])
d_ffn = d_ffn_choices[round(best_params[1])]
lr = best_params[2]


best_params_dict = {
    "n_layers": n_layers,
    "d_ffn": d_ffn,
    "lr": lr,
    "best_score": best_score
}
save_path = os.path.join(dir, 'best_params.json')# 保存最佳参数组合
with open(save_path, 'w') as f:
    json.dump(best_params_dict, f, indent=4)


history_save_path = os.path.join(dir, 'fitness_history.json')# 保存 fitness_history
with open(history_save_path, 'w') as f:
    json.dump(fitness_history, f, indent=4)
print(f"fitness_history save in {history_save_path}")


best_params_path = os.path.join(dir, 'best_params.json')
with open(best_params_path, 'r') as f:
    best_params = json.load(f)
n_layers = best_params["n_layers"]
d_ffn = best_params["d_ffn"]
lr = best_params["lr"]

# setup_seed(325)
lr_scheduler = StepLR(step_size=40000, gamma=1)
optimizer = AdamW(lr=lr, weight_decay=0.01)
saits = SAITS(n_steps=50, n_features=40, n_layers=n_layers, d_model=256, n_heads=4,
              d_k=64, d_v=64, d_ffn=d_ffn, dropout=0.1,epochs=500,
              optimizer=optimizer, batch_size=256,patience=40,
              model_saving_strategy='best',saving_path=dir, device="cuda:0")

start_time = time.time()
saits.fit(train_set, val_set)
end_time = time.time()
print(f"运行时间: {end_time - start_time}秒")

# Draw verification set
imputation,reconstruct = saits.impute(val_set)
scaler = joblib.load(os.path.join(dir, 'scaler.pkl'))
test = val_set['X']
test_ori = val_set['X_ori']
valmse = calc_mse(imputation,test_ori,np.isnan(test) ^ np.isnan(test_ori) )
print(f'valmse: {valmse}')
imputation = numpy.reshape(imputation, [imputation.shape[0], 2000, 1], order='F')
imputation_2d = imputation.reshape(imputation.shape[0]*imputation.shape[1],1)
imputation_2d = scaler.inverse_transform(imputation_2d)
imputation = imputation_2d.reshape(imputation.shape[0],imputation.shape[1],imputation.shape[2])
test = numpy.reshape(test, [test.shape[0], 2000, 1], order='F')
test_ori = numpy.reshape(test_ori, [test_ori.shape[0], 2000, 1], order='F')
indicating_mask = np.isnan(test) ^ np.isnan(test_ori)  # 用于计算插补误差的掩码矩阵
test_ori_2d = test_ori.reshape(test_ori.shape[0]*test_ori.shape[1],1)
test_ori_2d = scaler.inverse_transform(test_ori_2d)
test_ori = test_ori_2d.reshape(test_ori.shape[0],test_ori.shape[1],test_ori.shape[2])
rmse = np.sqrt(calc_mse(imputation, test_ori,indicating_mask))
print(f'RMSE: {rmse}')
miss = imputation[indicating_mask]
ori = test_ori[indicating_mask]
plt.figure()
plt.plot(miss, label='pre', color='red')
plt.plot(ori, label='targets', color='green')
plt.suptitle('Miss Value Prediction')
plt.show()


reconstruct = numpy.reshape(reconstruct, [reconstruct.shape[0], 2000, 1], order='F')
reconstruct_diff = np.diff(reconstruct,axis=1)
reconstruct_diff = np.concatenate([np.zeros((reconstruct_diff.shape[0], 1, reconstruct_diff.shape[2])), reconstruct_diff], axis=1)
reconstruct_diff_2d = reconstruct_diff.reshape(reconstruct_diff.shape[0]*reconstruct_diff.shape[1],1)
reconstruct_diff_2d = scaler.inverse_transform(reconstruct_diff_2d)
reconstruct_diff = reconstruct_diff_2d.reshape(reconstruct.shape[0],reconstruct.shape[1],reconstruct.shape[2])
reconstruct = np.cumsum(reconstruct_diff,axis=1)
rmse = np.sqrt(calc_mse(reconstruct, np.cumsum(test_ori,axis=1)))
print(f'RMSE: {rmse}')


