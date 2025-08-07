import numpy as np
import torch
import random

from torch import nn
from torch.utils.data import Dataset
import h5py
from sklearn.preprocessing import MinMaxScaler
import os

import h5py
import numpy as np


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def reset_seed():
    torch.manual_seed(torch.initial_seed())
    torch.cuda.manual_seed_all(torch.initial_seed())
    np.random.seed(None)  # 恢复为默认的随机状态
    random.seed(None)  # 使用系统时间等作为随机种子
    torch.backends.cudnn.deterministic = False  # 恢复非确定性行为
    torch.backends.cudnn.benchmark = True  # 恢复性能优化模式

def cpu(x):
    '''Transforms torch tensor into numpy array'''
    return x.cpu().detach().numpy()


class MMD_loss(nn.Module):
    def __init__(self, kernel_mul=2.0, kernel_num=5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassiankernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:

            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassiankernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num,
                                 fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY - YX)
        return loss


class HDF5BatchDataset(Dataset):
    def __init__(self, file_list, batch_size):
        self.file_list = file_list
        self.dataset_cache = {}
        self.batch_size = batch_size

    def __len__(self):
        return len(self.file_list) * self.batch_size  # Assuming each file contains 128 samples

    def __getitem__(self, idx):
        file_idx = idx // self.batch_size
        sample_idx = idx % self.batch_size

        if file_idx not in self.dataset_cache:
            if len(self.dataset_cache) >= 1:
                self.dataset_cache.clear()
                h5_file = self.file_list[file_idx]
                self.dataset_cache[file_idx] = h5_file
            h5_file = self.file_list[file_idx]
            with h5py.File(h5_file, 'r') as f:
                inputs = f['inputs'][:]
                targets = f['targets'][:]
            self.dataset_cache[file_idx] = (inputs, targets)

        inputs, targets = self.dataset_cache[file_idx]
        input_sample = inputs[sample_idx]
        target_sample = targets[sample_idx]

        return torch.tensor(input_sample, dtype=torch.float32), torch.tensor(target_sample, dtype=torch.float32)


def create_xy(dataset, n_past):
    data_x = []
    # data_y = []
    for i in range(n_past, len(dataset)+1):
        data_x.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        # data_y.append(dataset[i:i + 1, 0])
    return np.array(data_x)#, np.array(data_y)


def create_xy2(dataset, n_past,samplesize):
    data_x = []
    for i in range(samplesize):
        start_idx = np.random.randint(0, dataset.shape[0] - n_past + 1)
        data_x.append(dataset[start_idx:start_idx + n_past, :])
    return np.array(data_x)#, np.array(data_y)


# def create_xy(dataset, n_past):
#     data_x = []
#     data_y = []
#     scaler = MinMaxScaler(feature_range=(-1, 1))
#     for i in range(n_past, len(dataset)):
#         data_xt = scaler.fit_transform(dataset[i - n_past:i, 0:dataset.shape[1]])
#         data_yt = scaler.transform(dataset[i:i+1, 0:dataset.shape[1]])
#         data_yt = data_yt.reshape
#         data_x.append(data_xt)
#         data_y.append(data_yt)
#     return np.array(data_x), np.array(data_y)


def read_h5_dataset(filename):
    """
    读取 HDF5 文件中的特定数据集。

    参数:
    filename (str): HDF5 文件的路径。
    dataset_name (str): 要读取的数据集名称。

    返回:
    data (ndarray): 读取的数据集内容。
    """
    try:
        with h5py.File(filename, 'r') as f:
            data = f['data'][:]
            return data

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


class NPDCLoss(nn.Module):
    def __init__(self, n):
        """
        :param n: 滞后时间步的数量，用于计算过去n个时间点的平均值
        """
        super(NPDCLoss, self).__init__()
        self.n = n  # 时间滞后上限

    def forward(self, x_pred, x_true):
        ns = x_true.size(0)  # 样本数量

        # 初始化损失
        loss = 0.0

        for i in range(1, ns + 1):
            # 确保不会越界，最多取前 n 个时间点的均值
            delayed_values = []
            for j in range(1, self.n + 1):
                if i <= j:
                    delayed_values.append(x_true[i - 1].unsqueeze(0))
                else:
                    delayed_values.append(x_true[i - j - 1].unsqueeze(0))
            delayed_values = torch.cat(delayed_values, dim=0)

            delayed_avg = torch.mean(delayed_values, dim=0)
            # 计算每个样本的平方误差
            loss += (x_pred[i - 1] - delayed_avg) ** 2
        loss = loss / ns
        # loss = torch.log(loss + 1e-10)  # 避免 log(0) 的情况

        return loss


def parse_delta(masks):
    """
    根据输入的掩码（masks）和方向（dir_）计算 deltas，
    即每个时间步的增量值。主要功能是在处理时间序列数据时，通过掩码计算未被掩盖数据的时间间隔。
    """
    deltas = np.zeros((masks.shape[0], 20, 1))  # 初始化 delta

    for h in range(20):
        if h > 0:
            deltas[:, h, 0] = 1 + (1 - masks[:, h-1, 0]) * deltas[:, h - 1, 0]

    return deltas


def get_datasets_path(data_dir):
    train_set_path = os.path.join(data_dir, "train.h5")
    val_set_path = os.path.join(data_dir, "val.h5")

    with h5py.File(train_set_path, "r") as hf:
        train_X_arr = hf["X"][:]
        # train_X_Cumsum_arr = hf["X_Cumsum"][:]
    prepared_train_set = {"X": train_X_arr}

    if os.path.exists(val_set_path):
        with h5py.File(val_set_path, "r") as hf:
            val_X_arr = hf["X"][:]
            val_X_ori_arr = hf["X_ori"][:]
            # val_X_Cumsum_arr = hf["X_Cumsum"][:]

        prepared_val_set = {"X": val_X_arr, "X_ori": val_X_ori_arr}
        return prepared_train_set,prepared_val_set
    else:
        return prepared_train_set


def write_h5_dataset(out_dir, train_dataset, eval_dataset=None):
    train_output_file = os.path.join(out_dir, 'train.h5')
    with h5py.File(train_output_file, 'w') as h5f:
        # 创建 train 数据组并写入数据
        h5f.create_dataset('X', data=train_dataset['X'])
        h5f.create_dataset('X_Cumsum', data=train_dataset['X_Cumsum'])
    if eval_dataset is not None:
        eval_output_file = os.path.join(out_dir, 'val.h5')
        with h5py.File(eval_output_file, 'w') as h5f:
            h5f.create_dataset('X', data=eval_dataset['X'])
            h5f.create_dataset('X_ori', data=eval_dataset['X_ori'])
            h5f.create_dataset('X_Cumsum', data=eval_dataset['X_Cumsum'])

# def reconstruct(array,mask):
#     mask = mask.astype(int)
#     segments = np.diff(mask,axis=0)  # 计算 mask 的变化
#     boundaries = np.where(segments != 0)[0] + 1  # 找到变化的位置
#     boundaries = np.concatenate(([0], boundaries, [len(mask)]))  # 包括起点和终点
#     # 初始化结果数组
#     result = np.copy(array)
#     cumulative_sum = 0  # 累加值初始化
#     # 按段处理
#     for i in range(len(boundaries) - 1):
#         start, end = boundaries[i], boundaries[i + 1]
#         result[start:end] += cumulative_sum  # 当前段加上累加值
#         cumulative_sum = result[end - 1]  # 更新累加值为当前段的最后一个值
#     return result

def cumsum_with_nan_blocks(data):
    # 确保输入维度为 (n, 500, 1)
    assert len(data.shape) == 3 and data.shape[2] == 1, "输入数组应为 (n, 500, 1)"

    # 将形状展平为 (n, 500)
    data_2d = data[:, :, 0]
    n, rows = data_2d.shape

    # 初始化结果数组
    result = np.full_like(data_2d, np.nan)  # 与 data_2d 相同形状

    # 获取非NaN掩码
    mask = ~np.isnan(data_2d)

    # 创建分块索引
    block_indices = np.cumsum(~mask, axis=1) * mask  # 分块索引

    # 遍历每一行进行处理
    for i in range(n):
        unique_blocks = np.unique(block_indices[i][mask[i]])  # 获取当前行的有效块
        for block in unique_blocks:
            idx = block_indices[i] == block  # 当前块的掩码
            result[i, idx] = np.cumsum(data_2d[i, idx])  # 计算块内累积和

    # 将结果恢复为 (n, 500, 1)
    return result[:, :, np.newaxis]