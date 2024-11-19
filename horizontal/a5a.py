import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from horizontal.utils import *

params_for_plots = {'legend.fontsize': 'xx-large',
          'figure.figsize': (10, 8),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params_for_plots)

dataset = "datasets/a5a.txt"
data = load_svmlight_file(dataset)
X, Y = data[0].toarray(), data[1]

L = np.abs(np.max(np.linalg.eig(1 / X.shape[0] * np.dot(X.T, X))[0]))
mu = L / 100
L += mu

devices_num = 100
w0 = np.zeros(X.shape[1])

act_val = np.array(
 [-2.95007963e-01, -1.50701262e-01,  1.98616760e-02,  1.31844251e-01,
  5.77603713e-02, -1.35297219e-01, -7.01271412e-02,  5.78604281e-02,
  4.38271340e-02, -1.26255568e-02, -9.92627401e-03, -6.36487425e-04,
 -2.05973272e-04, -1.03389314e-01, -3.43131222e-02, -2.91496918e-02,
 -3.65554323e-02, -3.28353661e-02,  8.95506730e-02, -9.63060842e-02,
 -5.43394879e-02, -1.86334124e-01,  7.21611683e-02, -2.59458536e-02,
  2.43933077e-02, -3.92718161e-02, -4.87573886e-02, -1.49297400e-02,
  9.83233447e-02, -1.43254248e-02, -4.63947077e-02,  5.28264652e-02,
 -3.74628810e-02, -9.43037768e-03, -2.64911824e-01, -1.86334124e-01,
 -9.63060842e-02, -1.55254596e-03,  3.12861651e-01,  3.95053160e-01,
 -1.51313218e-01, -3.73444400e-01, -5.03362522e-02, -4.30183435e-02,
 -1.75607743e-02,  4.37690010e-03,  3.06832041e-02, -7.77366533e-02,
 -1.38442844e-01, -6.19828650e-03,  2.26856443e-01,  1.35788188e-01,
 -6.74682739e-02, -6.41986675e-02, -5.94382463e-02, -7.24250160e-02,
 -3.66328268e-02, -8.15000868e-03,  1.07387032e-02, -3.00832152e-04,
  1.37351281e-01, -2.20920961e-01,  2.69281421e-01, -2.14135537e-01,
 -5.71671261e-02, -1.50652005e-01, -8.68585806e-02, -1.70355078e-02,
 -2.89191568e-02, -2.14628348e-02, -8.19668469e-02, -2.21765754e-01,
 -1.44771730e-02, -4.06929533e-01,  1.70686606e-01, -3.12010458e-01,
  7.57675313e-02, -2.27507546e-01, -4.75373993e-02, -1.39607205e-01,
  4.53707498e-02,  1.33038474e-01, -1.14696945e-01,  7.08620116e-04,
 -4.00254630e-03, -1.32357065e-02,  2.45557639e-03, -7.58060684e-03,
 -1.40994783e-03, -8.10325955e-03,  4.05547251e-03,  2.24344873e-05,
 -1.68027218e-03, -4.56126479e-03,  1.69995542e-03, -9.65769977e-04,
 -1.88593256e-03,  9.11055333e-03,  6.79131087e-03, -2.24512803e-03,
  2.25645072e-03, -5.87005183e-03, -4.74437937e-02, -7.81037834e-04,
  8.98071860e-04,  3.27179769e-04, -5.80648515e-03, -1.44916036e-03,
  1.40752829e-04,  1.15906050e-03,  8.43081853e-04, -2.90892266e-03,
  8.75986972e-05, -3.20545341e-03, -1.38121314e-03,  8.71774832e-04,
 -3.85719041e-04,  3.03777476e-03, -1.18196692e-02, -1.24814523e-03,
 -3.34656447e-04,  6.35991982e-04])

init(L, mu, devices_num, w0, X, Y, act_val)

def generate_plot():
    N_GD = 1300
    x_GD, y_GD = GD(N_GD, 0.4)


    N_AGD = 200
    x_AGD, y_AGD = AGD(N_AGD)


    N_adiana = 3000
    x_adiana, y_adiana = ADIANA(N_adiana, 300)


    N_katyusha_hor_randK = 6000
    x_katyusha_hor_randK, y_katyusha_hor_randK = DHPL_Katyusha_Rand1divn(N_katyusha_hor_randK)


    N_katyusha_hor_permK = 6000
    x_katyusha_hor_permK, y_katyusha_hor_permK = DHPL_Katyusha_PermK(N_katyusha_hor_permK)


    num, limit_y, limit_x = 5, 5, 10000

    N_GD = x_GD[x_GD < limit_x].size
    N_AGD = x_AGD[np.log10(y_AGD) > -limit_y].size
    N_adiana = x_adiana[np.log10(y_adiana) > -limit_y].size
    N_katyusha_hor_randK = x_katyusha_hor_randK[np.log10(y_katyusha_hor_randK) > -limit_y].size
    N_katyusha_hor_permK = x_katyusha_hor_permK[np.log10(y_katyusha_hor_permK) > -limit_y].size

    plt.title('A5A')
    plt.ylabel(R"$\log \left ( |f(x^k) - f(x^*)| \; / \; |f(x^0 - f(x^*)| \right )$")
    plt.xlabel('Transfer units No.')
    plt.semilogy(x_GD[x_GD < limit_x], y_GD[x_GD < limit_x], markersize=10, markevery=int(N_GD/num), color='orange', marker='^', label='GD')
    plt.semilogy(x_AGD[np.log10(y_AGD) > -limit_y], y_AGD[np.log10(y_AGD) > -limit_y], markersize=10, markevery=int(N_AGD/num), color='red', marker='|', linestyle='-', linewidth=2, label='AGD')
    plt.semilogy(x_adiana[np.log10(y_adiana) > -limit_y], y_adiana[np.log10(y_adiana) > -limit_y], markersize=10, markevery=int(N_adiana/num), color='brown', marker='2', label='ADIANA')
    plt.semilogy(x_katyusha_hor_randK[np.log10(y_katyusha_hor_randK) > -limit_y], y_katyusha_hor_randK[np.log10(y_katyusha_hor_randK) > -limit_y], markersize=10, markevery= int(N_katyusha_hor_randK/num), color='green', marker='o', label='DHPL-Katyusha+RandK')
    plt.semilogy(x_katyusha_hor_permK[np.log10(y_katyusha_hor_permK) > -limit_y], y_katyusha_hor_permK[np.log10(y_katyusha_hor_permK) > -limit_y], markersize=10, markevery=int(N_katyusha_hor_permK/num), color='blue', marker='*', label='DHPL-Katyusha+PermK')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()