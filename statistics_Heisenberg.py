import msgpack
import glob
import numpy as np


def ratio_err_prop(x, y):
    x_ave = np.average(x)
    y_ave = np.average(y)
    cov = np.cov(x, y)
    x_var = cov[0, 0] / x.size
    y_var = cov[1, 1] / x.size
    xy_var = cov[0, 1] / x.size
    var = (x_var/(x_ave*x_ave) + y_var/(y_ave*y_ave) - 2.0*xy_var/(x_ave * y_ave))*np.square(x_ave/y_ave)
    return np.sqrt(var)


sample = glob.glob('./sample_*.mpac')[0]
data = msgpack.unpackb(open(sample, 'rb').read(), raw=True)
ene = np.array(data[b'sampled_energy'])
denom = np.array(data[b'sampled_denom'])
zz1 = np.array(data[b'sampled_zz1'])
zz2 = np.array(data[b'sampled_zz2'])
zz3 = np.array(data[b'sampled_zz3'])
xx1 = np.array(data[b'sampled_xx1'])
xx2 = np.array(data[b'sampled_xx2'])
xx3 = np.array(data[b'sampled_xx3'])
nbins_ini = 1
while denom.size // nbins_ini > 2048:
    nbins_ini *= 2

quantity = ['Energy', 'ZZ1', 'ZZ2', 'ZZ3', 'XX1', 'XX2', 'XX3']
data_list = [ene, zz1, zz2, zz3, xx1, xx2, xx3]
np.set_printoptions(legacy='1.25')
for label, raw_data in zip(quantity, data_list):
    print(label)
    print('Mean')
    print(np.average(raw_data) / np.average(denom))
    nbins = nbins_ini
    nbin_list = []
    error_list = []
    while denom.size // nbins >= 32:
        ndata = denom.size // nbins
        binned_data = np.empty(ndata)
        denom_bin = np.empty(ndata)
        for i in range(ndata):
            denom_bin[i] = np.average(denom[i*nbins:(i+1)*nbins])
            binned_data[i] = np.average(raw_data[i*nbins:(i+1)*nbins])
        nbin_list.append(nbins)
        error_list.append(ratio_err_prop(binned_data, denom_bin))
        nbins *= 2
    print('Bin size')
    print(nbin_list)
    print('Standard error')
    print(error_list)
