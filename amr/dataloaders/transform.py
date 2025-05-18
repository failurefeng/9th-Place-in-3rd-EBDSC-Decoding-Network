import numpy as np

__all__ = ['normalize_IQ', 'get_amp_phase', 'get_iq_framed']

def normalize_IQ(t):
    t_max = np.max(t)
    t_min = np.min(t)
    diff = t_max - t_min
    t_norm = (t - t_min) / diff
    return t_norm

def get_amp_phase(data):
    if data.ndim == 2:
        # 如果形状是 (2, L)，转换为 (1, 2, L)
        if data.shape[0] == 2:
            data = np.expand_dims(data, axis=0)  # 形状变为 (1, 2, L)
        # 如果形状是 (L, 2)，转换为 (1, 2, L)
        elif data.shape[1] == 2:
            data = np.expand_dims(data.T, axis=0)  # 转置并扩展维度，形状变为 (1, 2, L)
        else:
            raise ValueError(f"输入数据的形状 {data.shape} 不符合要求，应为 (2, L) 或 (L, 2)")
    signal_len = data.shape[-1]
    X_cmplx = data[:, 0, :] + 1j * data[:, 1, :]
    X_amp = np.abs(X_cmplx)
    X_ang = np.arctan2(data[:, 1, :], data[:, 0, :]) / np.pi

    X_amp_min = np.min(X_amp,1)
    X_amp_diff = np.max(X_amp,1) - np.min(X_amp,1)
    X_amp = ((X_amp.T - X_amp_min.T)/X_amp_diff.T).T

    X_amp = np.reshape(X_amp, (-1, 1, signal_len))
    X_ang = np.reshape(X_ang, (-1, 1, signal_len))
    X = np.concatenate((X_amp, X_ang), axis=1)
    return X

def get_iq_framed(X, L=32, R=16):
    # [2, 1024]
    F=int((X.shape[2]-L)/R+1)
    Y = np.zeros([F, X.shape[0], 2*L])
    i = 0
    for idx in range(0, X.shape[-1]-L+1, R):
        Y[i, :, :] = X[:, :, idx:idx+L].reshape([1,X.shape[0], 2*L])
        i = i+1
        #Y.append(X[:, :, idx:idx+L].reshape([1,X.shape[0], 2*L]))  # (2, L=32)
    #Y = np.vstack(Y)  # (F, 2L) = (63, 64)  F=(1024-L)/R+1
    Y = np.moveaxis(Y, 0, 1)
    return Y