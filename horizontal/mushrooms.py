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

dataset = "datasets/mushrooms.txt"
data = load_svmlight_file(dataset)
X, Y = data[0].toarray(), data[1]
Y = 2 * Y - 3

L = np.abs(np.max(np.linalg.eig(1 / X.shape[0] * np.dot(X.T, X))[0]))
mu = L / 100
L += mu

devices_num = 100
w0 = np.zeros(X.shape[1])

act_val = np.array(
   [-0.00299926,  0.06399174, -0.03243267, -0.02739051,  0.01400328, -0.01862826, -0.01506729, 
    -0.12403255, -0.00361259,  0.13925676,  0.01109306, -0.05826587, -0.01828119,  0.0202128,   
     0.05890176, -0.0431947,   0.01116186,  0.01116186, -0.00213216,  0.0058869,   0.19347757, 
    -0.19693324,  0.11350587, -0.11013653, -0.4170728,  -0.01636891,  0.11350587,  0.64656204, 
    -0.14908249, -0.09218436, -0.09218436,  0.04537265, -0.04882832, -0.22733108,  0.2238754,  
     0.39033699, -0.39379266, -0.25382928,  0.02570133, -0.03906527, -0.06228264,  0.04478202,  
     0.01835878,  0.12404779,  0.00984961, -0.02195577,  0.0520872,   0.09067491,  0.00817566, 
    -0.07067696,  0.06722129,  0.00163286,  0.24382427, -0.27585808,  0.02694527,  0.03001539,  
     0.16664225, -0.2547279,   0.05461459, -0.01636891, -0.06558184,  0.02537118,  0.08860682,  
     0.05379508, -0.05950111, -0.08154299,  0.05656084, -0.00479474, -0.01636891, -0.06518777,  
     0.02522226,  0.08822257,  0.05379508, -0.03742218, -0.08229254,  0.04117261, -0.01059678, 
    -0.00345567, -0.00479474,  0.02689754, -0.05245601,  0.02689754,  0.07681304, -0.0638998, 
    -0.01636891,  0.21453451, -0.01636891, -0.03897832, -0.19618929,  0.03354633,  0.01398285, 
    -0.31406591,  0.20730508,  0.01398285,  0.2293671,  -0.06633388,  0.02026017, -0.12193679,  
     0.01398285,  0.07375262,  0.04516244,  0.10136603, -0.00358636, -0.29644335,  0.07629295,  
     0.06016033,  0.06428245,  0.03416438, -0.01429323, -0.09693288, -0.10096949,  0.05013277])

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


    N_katyusha_hor_permK = 4000
    x_katyusha_hor_permK, y_katyusha_hor_permK = DHPL_Katyusha_PermK(N_katyusha_hor_permK)


    num, limit_y, limit_x = 5, 5, 6000

    N_GD = x_GD[x_GD < limit_x].size
    N_AGD = x_AGD[np.log10(y_AGD) > -limit_y].size
    N_adiana = x_adiana[np.log10(y_adiana) > -limit_y].size
    N_katyusha_hor_randK = x_katyusha_hor_randK[np.log10(y_katyusha_hor_randK) > -limit_y].size
    N_katyusha_hor_permK = x_katyusha_hor_permK[np.log10(y_katyusha_hor_permK) > -limit_y].size

    plt.title('MUSHROOMS')
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