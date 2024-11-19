import numpy as np
import matplotlib.pyplot as plt
from utils import *
from sklearn.datasets import load_svmlight_file

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

L = np.abs(np.max(np.linalg.eig(2 / X.shape[0] * np.dot(X.T, X))[0]))
mu = L / 100
L += mu

devices_num = 40
w0 = np.zeros(X.shape[1])

act_val = np.array([
 -1.23335596e-01, -1.16558097e-01,  3.95847851e-03,  8.59081269e-02,
  4.16247300e-02, -4.16136630e-03, -7.77487048e-02,  5.75375602e-02,
  6.34465700e-02, -1.56878977e-02, -1.16632240e-02, -1.62562445e-03,
  2.38463457e-04, -6.54179026e-02, -1.50083060e-02, -1.67240552e-02,
 -7.43649901e-03, -3.81559408e-03, -1.04066902e-02, -4.20999321e-02,
 -1.68948692e-02, -8.17653943e-02,  8.58438220e-02, -3.89487107e-02,
  3.00687230e-02, -3.49210676e-02, -4.86828199e-02,  3.58873859e-03,
  6.23353624e-02, -1.28697221e-02, -1.95912675e-02,  6.85757773e-02,
 -4.19585771e-02, -1.06757294e-02, -1.82005314e-01, -8.17653943e-02,
 -4.20999321e-02, -8.87998772e-03,  2.06348271e-01,  2.15834227e-01,
 -1.03514034e-01, -1.22097513e-01, -4.70224394e-02, -4.46330125e-02,
 -1.47101073e-02,  7.74052202e-03,  4.57738755e-02, -6.00959617e-02,
 -6.00760808e-02,  1.21803679e-02,  1.86704469e-01,  9.01899671e-02,
 -4.19095528e-02, -4.97228901e-02, -9.13468412e-03, -8.24802552e-02,
 -4.04326298e-02, -4.54543350e-03,  2.40602827e-02, -4.14161321e-04,
  1.38543204e-01, -8.03739962e-02,  9.98179162e-02, -1.13823372e-01,
 -4.27748593e-02, -1.09791249e-01, -5.27375301e-03, -8.96411715e-03,
 -4.17527878e-02, -2.57369836e-02, -2.66747154e-02, -8.72714504e-02,
 -2.11309066e-02, -2.36956075e-01,  1.28553718e-01, -1.80235304e-01,
  7.18329469e-02, -1.17403710e-01, -3.06421614e-02, -6.23549162e-02,
  3.14800211e-02,  7.05184091e-02, -6.97801888e-04,  2.33550502e-03,
 -7.01308898e-03, -1.78039627e-02,  7.53366828e-03, -1.18508642e-02,
 -1.72973949e-03, -1.52437538e-02,  9.33977768e-03,  2.17082201e-03,
 -1.05468453e-04, -8.24635494e-03,  3.90199193e-03, -1.18583520e-03,
 -5.59762239e-04,  2.15845406e-02,  1.36040193e-02, -1.78968711e-03,
  6.39732630e-03, -7.13694269e-03, -4.66911912e-02, -1.24013537e-03,
  2.94097729e-03,  9.38908476e-04, -5.27987054e-03, -1.77436199e-03,
 -3.32256549e-05,  9.85052141e-04,  4.30615010e-03, -2.81964900e-03,
  2.82106981e-04,  4.85027559e-04, -9.27025103e-04,  9.06865948e-04,
  1.27137827e-06,  6.24088883e-03, -1.22860590e-02, -2.94942685e-03,
 -5.09952511e-04,  2.34952346e-03])

init(L, mu, devices_num, w0, X, Y, act_val)

def generate_plot():
    N_GD = 100
    x_GD, y_GD = GD(N_GD, 0.7)
    x_GD /= 1000


    N_AGD = 100
    x_AGD, y_AGD = AGD(N_AGD)
    x_AGD /= 1000


    N_rand_alg2 = 900
    x_rand_alg2, y_rand_alg2 = DVPL_Katyusha_RandKAlg2(N_rand_alg2, 1/devices_num)
    x_rand_alg2 /= 1000


    N_rand_alg3 = 1100
    x_rand_alg3, y_rand_alg3 = DVPL_Katyusha_RandKAlg3(N_rand_alg3, 1/devices_num, 1.1)
    x_rand_alg3 /= 1000


    N_perm = 1500
    x_perm, y_perm = DVPL_Katyusha_PermK(N_perm, 1.1)
    x_perm /= 1000

    num, limit_y, limit_x = 5, 5, 800

    N_GD = x_GD[x_GD < limit_x].size
    N_AGD = x_AGD[np.log10(y_AGD) > -limit_y].size
    N_rand_alg2 = x_rand_alg2[np.log10(y_rand_alg2) > -limit_y].size
    N_rand_alg3 = x_rand_alg3[np.log10(y_rand_alg3) > -limit_y].size
    N_perm = x_perm[np.log10(y_perm) > -limit_y].size

    plt.title('A5A')
    plt.ylabel(R"$\log \left ( |f(x^k) - f(x^*)| \; / \; |f(x^0 - f(x^*)| \right )$")
    plt.xlabel('Transfer units No. (in thousands)')
    plt.semilogy(x_GD[x_GD < limit_x], y_GD[x_GD < limit_x], markersize=10, markevery=int(N_GD/num), color='orange', marker='^', label='GD')
    plt.semilogy(x_AGD[np.log10(y_AGD) > -limit_y], y_AGD[np.log10(y_AGD) > -limit_y], markersize=10, markevery=int(N_AGD/num), color='red', marker='|', linestyle='-', linewidth=2, label='AGD')
    plt.semilogy(x_rand_alg2[np.log10(y_rand_alg2) > -limit_y], y_rand_alg2[np.log10(y_rand_alg2) > -limit_y], markersize=10, markevery=int(N_rand_alg2/num), color='brown', marker='2', label='DVPL-Katyusha(Alg.2)+RandK')
    plt.semilogy(x_rand_alg3[np.log10(y_rand_alg3) > -limit_y], y_rand_alg3[np.log10(y_rand_alg3) > -limit_y], markersize=10, markevery=int(N_rand_alg3/num), color='green', marker='2', label='DVPL-Katyusha(Alg.3)+RandK')
    plt.semilogy(x_perm[np.log10(y_perm) > -limit_y], y_perm[np.log10(y_perm) > -limit_y], markersize=10, markevery=int(N_perm/num), color='blue', marker='2', label='DVPL-Katyusha(Alg.3)+PermK')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()