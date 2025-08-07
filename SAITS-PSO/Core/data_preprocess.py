import matplotlib.pyplot as plt

from Core import *
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from pygrinder import mcar,seq_missing,mnar_x,mnar_t
from sklearn.model_selection import train_test_split
from pypots.data import load_specific_dataset
import joblib
import scipy

if __name__ == "__main__":
    # '小球数据制作训练数据'
    # filename = r'H:\DriftCorrection\Missing data filling\data\小球数据\inputs.h5'
    # out_dir = r'H:\DriftCorrection\Missing data filling\data\小球数据\z2000'
    # seq_len = 2000
    # miss_rate = 0.5
    # dataset = read_h5_dataset(filename)
    # dataset = np.transpose(dataset[3:4, :])
    # scaler = StandardScaler()
    # dataset = scaler.fit_transform(dataset)
    # X = create_xy(dataset, seq_len)
    #
    # train_size = int(X.shape[0] * 0.8)
    # train_X = X[0:train_size, :]
    # eval_X_ori = X[train_size:X.shape[0], :]
    # np.random.shuffle(train_X)
    # np.random.shuffle(eval_X_ori)
    # train_X = mcar(train_X, miss_rate)
    #
    # nan_ratio = np.isnan(train_X).sum() / train_X.size
    # print(f"NaN的比例是: {nan_ratio}")
    #
    # eval_X = mcar(eval_X_ori, miss_rate)
    # train_dataset = {"X": train_X}  # X用于模型输入
    # eval_dataset = {"X": eval_X, "X_ori": eval_X_ori}
    # write_h5_dataset(out_dir, train_dataset, eval_dataset)
    # joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))

    '真实数据制作训练数据'
    datapath = r'Y:\宋启航\data\Figure6Data\R4\LS\470'  # 效果不行的话制作数据时换成顺序划分33-37行
    with h5py.File(os.path.join(datapath, 'Drift_diff.mat'), 'r') as f:
        dataset = np.array(f['Drift_diff'])
    out_dir = r'Y:\宋启航\data\实验数据for新网络\X'
    seq_len = 2000
    miss_rate = 0.5
    scaler = StandardScaler()
    # dataset = data['Drift_diff']
    dataset = dataset[1:2, :, :]
    dataset = np.where(dataset == 0, np.nan, dataset)
    d1, d2, d3 = dataset.shape
    dataset_s = dataset.reshape(dataset.shape[1]*dataset.shape[2],-1)
    dataset_s = scaler.fit_transform(dataset_s)
    dataset = dataset_s.reshape(d1, d2, d3)
    train_X = []
    train_X_Cumsum = []
    eval_X = []
    eval_X_ori = []
    for i in range(dataset.shape[2]):
        datasetT = dataset[0:1, :,i]
        datasetT = np.transpose(datasetT)
        # datasetT = np.where(datasetT == 0, np.nan, datasetT)
        # datasetT = scaler.fit_transform(datasetT)
        X = create_xy(datasetT, seq_len)
        train_size = int(X.shape[0] * 0.8)
        train_XT = X[0:train_size, :]
        eval_X_oriT = X[train_size:X.shape[0], :]
        np.random.shuffle(train_XT)
        np.random.shuffle(eval_X_oriT)
        # train_X_oriT, eval_X_oriT = train_test_split(X, test_size=0.2, random_state=42)
        train_X_Cumsum.append(train_XT)
        train_XT = mcar(train_XT, miss_rate)
        nan_ratio = np.isnan(train_XT).sum() / train_XT.size
        print(f"NaN的比例是: {nan_ratio}")
        # train_X = seq_missing(train_X, 0.05, miss_len, [0], loc)
        eval_XT = mcar(eval_X_oriT, miss_rate)
        train_X.append(train_XT)
        eval_X.append(eval_XT)
        eval_X_ori.append(eval_X_oriT)

    train_X_Cumsum = np.concatenate(train_X_Cumsum, axis=0)
    train_X_Cumsum = np.cumsum(train_X_Cumsum,axis=1)
    train_X = np.concatenate(train_X, axis=0)
    eval_X = np.concatenate(eval_X, axis=0)
    eval_X_ori = np.concatenate(eval_X_ori, axis=0)
    shuffle_ids = torch.randperm(train_X.shape[0])
    train_X = train_X[shuffle_ids]
    train_X_Cumsum = train_X_Cumsum[shuffle_ids]
    shuffle_ids = torch.randperm(eval_X.shape[0])
    eval_X = eval_X[shuffle_ids]
    eval_X_ori = eval_X_ori[shuffle_ids]
    eval_X_Cumsum = np.cumsum(eval_X_ori,axis=1)
    train_dataset = {"X": train_X,"X_Cumsum":train_X_Cumsum}  # X用于模型输入
    eval_dataset = {"X": eval_X, "X_ori": eval_X_ori,"X_Cumsum":eval_X_Cumsum}
    write_h5_dataset(out_dir, train_dataset, eval_dataset)
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.pkl'))