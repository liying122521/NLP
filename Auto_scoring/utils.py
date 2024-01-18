
import scipy
import torch
import numpy as np
import scipy.stats
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


def l2_normalize(vecs):
    """用于对向量进行标准化（L2 归一化）
    """
    norms = torch.norm(vecs, p=2, dim=1, keepdim=True)  # todo
    return vecs / torch.clamp(norms, min=1e-8) # todo
    # norms = (vecs**2).sum(axis=1, keepdims=True)**0.5

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def compute_pearsonr(x, y):
    # compute_pearsonr(x, y) 函数使用的是 Pearson 相关系数。Pearson 相关系数用于衡量两个变量之间的线性关系。它基于两个变量的协方差和标准差的比率来计算。
    # 输出:(r, p)
    # r:相关系数[-1，1]之间
    # p:相关系数显著性
    # 所有下面的数据选第零位
    return scipy.stats.pearsonr(x, y)[0]

def compute_mse(all_preds, all_labels):
    mse = mean_squared_error(all_labels, all_preds)
    return mse
def compute_mae(all_preds, all_labels):
    mae = mean_absolute_error(all_preds, all_labels)
    return mae

def compute_rmse(all_preds, all_labels):
    mse = mean_squared_error(all_preds, all_labels)
    rmse = np.sqrt(mse)
    return rmse
