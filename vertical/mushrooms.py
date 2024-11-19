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

dataset = "datasets/mushrooms.txt"
data = load_svmlight_file(dataset)
X, Y = data[0].toarray(), data[1]
Y = 2 * Y - 3

L = np.abs(np.max(np.linalg.eig(2 / X.shape[0] * np.dot(X.T, X))[0]))
mu = L / 100
L += mu

devices_num = 40
w0 = np.zeros(X.shape[1])

act_val = np.array([-0.00683757,  0.03863214, -0.03349889, -0.00441723,  0.02841602,
                    -0.02461848,  0.01385955, -0.06667231, -0.00749594,  0.0579847,
                     0.01394006, -0.07861355,  0.01268968,  0.00641401,  0.03579437,
                    -0.06983339,  0.02195447,  0.02195447, -0.0211121,   0.05448798,
                     0.06870498, -0.07102899,  0.1754774,  -0.15801321, -0.25840441,
                    -0.0328316,   0.1754774,   0.4457601,  -0.19089097, -0.07944936,
                    -0.07944936,  0.02951854, -0.03184255, -0.14529995,  0.14297594,
                     0.24137347, -0.24369748, -0.11569841,  0.02931375, -0.01424294,
                    -0.03794029,  0.03463043,  0.01623496,  0.06444104,  0.00307179,
                    -0.04029602,  0.02215443,  0.04022776, -0.00422053, -0.01392088,
                     0.01159687, -0.00297636,  0.09533932, -0.09196711, -0.00271986,
                     0.01346879,  0.02963232, -0.07665986,  0.03123474, -0.0328316,
                    -0.03540368,  0.02784173,  0.03619439,  0.04659143, -0.02920709,
                    -0.01613343,  0.01298202, -0.01235779, -0.0328316,  -0.0328065,
                     0.02750499,  0.03515849,  0.04659143, -0.00847996, -0.01650797,
                     0.00675135, -0.02770425, -0.00232401, -0.01235779,  0.02329571,
                    -0.03655765,  0.02329571,  0.02855711,  0.00195048, -0.0328316,
                     0.07240217, -0.0328316,   0.01072336, -0.11930107,  0.06668312,
                     0.01357748, -0.2349216,   0.13546432,  0.01357748,  0.14872945,
                    -0.1239309,   0.04299268, -0.01139041,  0.01357748,  0.02226082,
                     0.0126292,   0.10055048, -0.03195867, -0.12330242,  0.01749657,
                     0.00496268, -0.01186179,  0.026676,    0.01552006, -0.00988318,
                    -0.08511804,  0.05738027])

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
    x_rand_alg3, y_rand_alg3 = DVPL_Katyusha_RandKAlg3(N_rand_alg3, 1/devices_num, 1.5)
    x_rand_alg3 /= 1000


    N_perm = 1500
    x_perm, y_perm = DVPL_Katyusha_PermK(N_perm, 1.5)
    x_perm /= 1000

    num, limit_y, limit_x = 5, 5, 800

    N_GD = x_GD[x_GD < limit_x].size
    N_AGD = x_AGD[np.log10(y_AGD) > -limit_y].size
    N_rand_alg2 = x_rand_alg2[np.log10(y_rand_alg2) > -limit_y].size
    N_rand_alg3 = x_rand_alg3[np.log10(y_rand_alg3) > -limit_y].size
    N_perm = x_perm[np.log10(y_perm) > -limit_y].size

    plt.title('MUSHROOMS')
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