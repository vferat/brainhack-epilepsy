import numpy as np
from pywt import wavedec
from joblib import Parallel, delayed
from scipy.signal import butter, sosfilt
from scipy import stats
from scipy.fft import dct
from antropy import spectral_entropy, svd_entropy


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply bandpass filter (Butterworth).
    """
    sos = butter(N=order, Wn=[lowcut, highcut], analog=False, btype='bandpass', fs=fs, output='sos')
    return sosfilt(sos, data, axis=-1)


def activity(epochs):
    """ Compute Hjorth parameter: Activity"""
    return np.var(epochs, axis=-1)


def mobility(epochs):
    """ Compute Hjorth parameter: Mobility """
    return np.sqrt(
        np.var(np.gradient(epochs, axis=-1), axis=-1) / activity(epochs)
    )


def complexity(epochs):
    """ Compute Hjorth parameter: Complexity """
    return mobility(np.gradient(epochs, axis=-1)) / mobility(epochs)


def extract_time_feat(X):
    """
    Create time-domain features from array of data points.
    Features include for each epoch:
        * four first statistical moments (mean, variance, skewness, kurtosis) for epoch signal and its two first derivatives and signal after Nonlinear Teager-Kaiser Energy Transform,
        * maximum absolute value,
        * curve length
        * two of Hjorth parameters: mobility and complexity (activity is included in stat moments)
    :param X: Data extracted from epochs for one or several subjects. Shape (n_epochs, n_channels, n_times).
    :type X: numpy.ndarray
    :return: Array with time features. Shape (n_epochs, n_channels, n_time_features=4).
    :rtype: numpy.ndarray
    """
    Xf = X.copy()

    t_feat = np.empty((Xf.shape[0], Xf.shape[1], 14))

    # stat moments of epochs
    t_feat[:, :, 0] = np.mean(Xf, axis=-1)
    t_feat[:, :, 1] = np.var(Xf, axis=-1)
    t_feat[:, :, 2] = stats.skew(Xf, axis=-1)
    t_feat[:, :, 3] = stats.kurtosis(Xf, axis=-1)

    # stat moments of derivative
    grad = np.gradient(Xf, axis=-1)
    t_feat[:, :, 4] = np.mean(grad, axis=-1)
    t_feat[:, :, 5] = np.var(grad, axis=-1)
    t_feat[:, :, 6] = stats.skew(grad, axis=-1)
    t_feat[:, :, 7] = stats.kurtosis(grad, axis=-1)

    # stat moments of second derivative
    grad = np.gradient(grad, axis=-1)
    t_feat[:, :, 8] = np.mean(grad, axis=-1)
    t_feat[:, :, 9] = np.var(grad, axis=-1)
    t_feat[:, :, 10] = stats.skew(grad, axis=-1)
    t_feat[:, :, 11] = stats.kurtosis(grad, axis=-1)

    # Hjorth parameters
    t_feat[:, :, 12] = mobility(Xf)
    t_feat[:, :, 13] = complexity(Xf)

    return t_feat


def extract_freq_feat(X, sfreq):
    """
    Compute frequency features, that are energies of signal within different frequency bands.
    :param sfreq: Sampling frequency of the EEG signal.
    :type sfreq: float
    :param X: Array with epochs data. Shape (n_epochs, n_channels, n_times).
    :type X: numpy.ndarray
    :return: Array with freq features. Shape (n_epochs, n_channels, len(freq_bands)).
    :rtype: numpy.ndarray
    """

    freq_bands = [(0.1, 3.5), (3.5, 8), (8, 12), (12, 15), (15, 18), (18, 30)]
    n_bands = len(freq_bands)
    n_times = X.shape[2]

    f_feat = np.empty((X.shape[0], X.shape[1], n_bands + 2))

    for i_fb, fb in enumerate(freq_bands):
        f_feat[:, :, i_fb] = np.sum(butter_bandpass_filter(X, fb[0], fb[1], sfreq, order=3) ** 2, axis=-1) / n_times

    # spectral edge frequency
    Xdct = dct(X.copy(), type=2, n=None, axis=- 1, norm='ortho', overwrite_x=False, workers=1)
    f_feat[:, :, n_bands + 0] = np.percentile(Xdct, 80, overwrite_input=False)
    f_feat[:, :, n_bands + 1] = np.percentile(Xdct, 95, overwrite_input=False)
    del Xdct

    return f_feat


def extract_information_feat(X, sfreq, svd_order=3, svd_delay=1, svd_norm=True, n_jobs=1):
    """
    Compute information features:
        * SVD entropy
        * Spectral entropy
    :param X: Array with epochs data. Shape (n_epochs, n_channels, n_times)
    :type X: numpy.ndarray
    :param sfreq: Sampling frequency
    :type sfreq: float
    :param svd_order: Order of SVD entropy (= length of the embedding dimension)
    :type svd_order: int
    :param svd_delay: Time delay (lag)
    :type svd_delay: int
    :param svd_norm: If True, divide by log2(order!) to normalize the entropy between 0 and 1. Otherwise, return the permutation entropy in bit
    :type svd_norm: bool
    :param n_jobs: number of workers. joblib.Parallel parameter.
    :type n_jobs: int
    :return: Features array, shape (n_epochs, n_channels, 1 SVD entropy + 1 Spectral entropy = 2)
    :rtype: numpy.ndarray
    """

    def process(x_):
        svde = []
        for i_ch in range(X.shape[1]):
            svde.append(np.expand_dims(svd_entropy(x_[i_ch], svd_order, svd_delay, normalize=svd_norm), axis=0))
        return np.concatenate(svde, axis=0)

    inf_feat = np.empty((X.shape[0], X.shape[1], 2))

    # svd entropy
    inf_feat[:, :, 0] = Parallel(n_jobs=n_jobs)(delayed(process)(x) for x in X)

    # spectral entropy
    inf_feat[:, :, 1] = spectral_entropy(X.copy(), sf=sfreq, method='welch', normalize=True, axis=-1)
    return inf_feat


def extract_dwt_feat(X):
    """
    Compute statistics from Daubechies 4 WT coefficients. Levels that are used: A5, D5, D4, D3.
    :param X: numpy array with epochs data, shape (n_epochs, n_channels, n_times)
    :return: numpy array with shape (n_epochs, n_channels, n_levels*n_statistics), where n_levels=4, n_statistics=4.
    """

    Xf_ = []
    # Compute A5 level features
    Xdwt = wavedec(X, 'db4', mode='periodic', level=5)[:5]
    Xf_.append(np.mean(Xdwt[0], axis=-1))
    Xf_.append(np.mean(Xdwt[0] * Xdwt[0], axis=-1))
    Xf_.append(stats.skew(Xdwt[0], axis=-1))
    Xf_.append(stats.kurtosis(Xdwt[0], axis=-1))
    Xf_.append(
        np.mean(np.abs(Xdwt[0]), axis=-1) / (np.mean(np.abs(Xdwt[0]), axis=-1) + np.mean(np.abs(Xdwt[1]), axis=-1)))
    # Compute D5-D3 levels' features
    for i_lev in [1, 2, 3]:
        Xf_.append(np.mean(Xdwt[i_lev], axis=-1))
        Xf_.append(np.mean(Xdwt[i_lev] * Xdwt[i_lev], axis=-1))
        Xf_.append(stats.skew(Xdwt[i_lev], axis=-1))
        Xf_.append(stats.kurtosis(Xdwt[i_lev], axis=-1))
        Xf_.append(np.mean(np.abs(Xdwt[i_lev]), axis=-1) / (
                np.mean(np.abs(Xdwt[i_lev]), axis=-1) +
                np.mean(np.abs(Xdwt[i_lev + 1]), axis=-1) +
                np.mean(np.abs(Xdwt[i_lev - 1]), axis=-1))
                   )
    return np.stack(Xf_, axis=2)