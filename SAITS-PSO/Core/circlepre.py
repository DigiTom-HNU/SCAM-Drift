from scipy.io import savemat

from Core import *
import joblib
import json
from pygrinder import mcar
from pypots.imputation import SAITS
import scipy
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import numpy
"After interpolating x, when interpolating y and z, you need to modify Dim dir to change it to the datapath for y or z." \
" Add 'imputation' after the datapath. The way of reading mat files needs to be modified. The last line for saving also " \
"needs to be modified: comment out lines 74-78 and uncomment line 80."
if __name__ == "__main__":
    Dim = slice(3, 4)
    filename = 'Drift_diff.mat'
    current_dir = os.path.dirname(__file__)  # Core/
    parent_dir = os.path.dirname(current_dir)  # 项目根目录/
    dir = os.path.join(parent_dir, 'Z')  # 项目根目录/X/
    datapath = os.path.join(parent_dir, '470\imputation')
    scaler = StandardScaler()#joblib.load(os.path.join(dir, 'scaler.pkl'))
    model_path = os.path.join(dir, 'SAITS.pypots')
    best_params_path = os.path.join(dir, 'best_params.json')
    # with open(best_params_path, 'r') as f:
    #     best_params = json.load(f)
    # n_layers = best_params["n_layers"]
    # lr = best_params["lr"]
    # d_ffn = best_params["d_ffn"]
    saits = SAITS(n_steps=50, n_features=40, n_layers=2, d_model=256, n_heads=4, d_k=64, d_v=64, d_ffn=512)
    saits.load(model_path)  # 你随时可以重新加载保存的模型文件以进行后续的插补或训练
    # with h5py.File(os.path.join(datapath, filename), 'r') as f:
    #     dataset = np.array(f['Drift_diff'])
    # dataset = np.transpose(dataset, (2, 1, 0))
    data = scipy.io.loadmat(os.path.join(datapath,filename))
    dataset = data['Drift_diff']
    k = 2000
    segments = []
    dataset_correct = dataset.copy()
    for i in range(dataset.shape[0]):
        datasetT = dataset[i,:,Dim]
        datasetT = np.where(datasetT == 0, np.nan, datasetT)
        datasetT = scaler.fit_transform(datasetT)
        for j in range(0, datasetT.shape[0] - 2000 + 1, k):
            segment = datasetT[j:j + 2000, :]
            mask = np.isnan(segment)
            if j == 0:
                mask[0] = False
            segment = numpy.reshape(segment, [50, 40], order='F')
            segment = segment.reshape(1, -1, 40)
            testset = {'X': segment}
            res = saits.predict(testset)
            pre = res['imputation']
            pre = pre[0, :, :]
            pre = numpy.reshape(pre, [2000, 1], order='F')
            pre = scaler.inverse_transform(pre)
            dataset_correct[i, j:j + 2000, Dim][mask] = pre[mask]
            # print(segment)
        # Ensure to handle the last slice, which may be less than 500 elements
        if datasetT.shape[0] - 2000 > 0:
            last_segment = datasetT[datasetT.shape[0] - 2000: datasetT.shape[0]]
            mask = np.isnan(last_segment)
            mask[0] = False
            last_segment = numpy.reshape(last_segment, [50, 40], order='F')
            last_segment = last_segment.reshape(1, -1, 40)
            testset = {'X': last_segment}
            res = saits.predict(testset)
            pre = res['imputation']
            pre = pre[0, :, :]
            pre = numpy.reshape(pre, [2000, 1], order='F')
            pre = scaler.inverse_transform(pre)
            dataset_correct[i, datasetT.shape[0] - 2000: datasetT.shape[0], Dim][mask] = pre[mask]


    Drift_diff = {
        "Drift_diff": dataset_correct,  # 单个值
    }

    # imputation_path = os.path.join(datapath, 'imputation')
    # if not os.path.exists(imputation_path):
    #     os.makedirs(imputation_path)
    # savemat(os.path.join(datapath,'imputation',filename), Drift_diff)

    savemat(os.path.join(datapath,filename), Drift_diff)