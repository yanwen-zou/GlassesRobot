import numpy as np

def umeyama(X, Y, estimate_scale=True):
    """
    Args:
        X: (N, 3) 源点
        Y: (N, 3) 目标点
        estimate_scale: 是否估计尺度（Sim3）。False 时是 SE3。

    Returns:
        s: 尺度
        R: (3,3) 旋转矩阵
        t: (3,) 平移向量
    """
    assert X.shape == Y.shape
    N = X.shape[0]

    mean_X = X.mean(axis=0)
    mean_Y = Y.mean(axis=0)

    Xc = X - mean_X
    Yc = Y - mean_Y

    # 协方差矩阵
    Sigma = (Yc.T @ Xc) / N

    # SVD
    U, D, Vt = np.linalg.svd(Sigma)

    # 处理反射，保证 det(R)=+1
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt

    if estimate_scale:
        var_X = (Xc ** 2).sum() / N
        s = (D @ np.diag(S)).trace() / var_X
    else:
        s = 1.0

    t = mean_Y - s * R @ mean_X

    return s, R, t
