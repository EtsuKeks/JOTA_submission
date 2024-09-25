import gc
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

params_for_plots = {'legend.fontsize': 'xx-large',
          'figure.figsize': (3, 4),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params_for_plots)

dataset = "w3a.txt"
data = load_svmlight_file(dataset)
X_train, Y_train = data[0].toarray(), data[1]

L = np.abs(np.max(np.linalg.eig(2/X_train.shape[0] * np.dot(X_train.T, X_train))[0]))
mu = L / 100
L += mu

devices_num = 100
batch_size = 1_000
K = 5
omega = 1/K * 100

act_val = [-7.24470057e-02, -9.46654952e-02, -1.37249929e-01, -1.54101031e-01,
 -1.08264728e-01, -8.20485589e-02, -3.18537247e-02, -2.14441498e-02,
 -2.58271399e-02, -8.22996702e-02, -1.08669735e-01, -3.61935782e-03,
 -1.66399764e-02, -6.21032121e-02, -3.75243977e-02, -1.59617170e-01,
 -7.60535537e-02, -2.40915010e-02, -1.19621667e-01, -8.41207053e-02,
 -1.71425644e-01, -2.42474164e-03, -1.13078300e-01, -8.87248999e-02,
 -3.13233477e-02, -2.46209175e-02, -2.76949821e-02, -2.81130358e-02,
 -2.20612209e-02, -2.74303101e-02, -1.63775684e-02, -3.83803257e-02,
 -8.38133036e-03, -2.36592744e-02, -1.78777378e-01, -5.59345345e-03,
 -9.62612873e-02, -1.20836975e-01,  3.23721415e-02,  0.00000000e+00,
 -2.91847556e-02,  2.76948298e-02,  2.76948298e-02, -5.58814302e-02,
  4.13640332e-02, -1.66554297e-03, -1.34954651e-02,  1.88138868e-02,
  7.78675354e-03,  5.80393671e-02,  3.33047718e-02, -1.66554297e-03,
  1.06064529e-02, -4.24070505e-02, -1.56519675e-01,  1.13005381e-02,
 -3.25640391e-03,  2.90096226e-02,  1.41614078e-02, -1.66421887e-02,
  2.27945382e-02, -4.17491772e-02,  3.98231094e-02,  1.88989592e-02,
  6.06116950e-02,  3.65257943e-02,  2.56563846e-02, -9.24923773e-03,
  2.17357244e-04, -5.00142157e-02, -1.29798393e-02,  1.05739308e-02,
  5.81427918e-03,  3.37068120e-02, -7.83174441e-02,  1.20761582e-01,
  7.37580261e-02, -2.98803299e-02, -9.03720016e-02, -1.04196419e-01,
 -4.95056320e-02,  7.87431437e-03,  2.46701896e-02, -4.69489366e-02,
 -3.54436885e-02, -6.68968789e-03, -8.33140584e-02,  1.07099019e-03,
 -4.08225629e-02, -1.92716122e-02,  3.61349954e-02,  6.21946450e-02,
  6.70844306e-02, -1.15996279e-03,  5.12185118e-02,  2.91394768e-02,
  5.81587993e-02,  1.72769767e-03, -4.56213635e-02,  2.18712254e-02,
 -6.76263675e-02,  1.27221533e-02, -3.67465599e-02,  5.13896612e-03,
 -1.14780758e-02, -1.12035282e-02, -1.44314052e-02,  1.63660997e-04,
 -1.96851180e-02, -1.66554297e-03,  2.76455467e-02, -3.84360988e-03,
  1.80000709e-02, -4.31985027e-02, -9.62255648e-02,  3.88907548e-02,
 -2.56390118e-01, -6.01772204e-02, -5.28902644e-02, -1.66554297e-03,
 -1.66554297e-03, -1.66554297e-03, -1.66554297e-03, -1.66554297e-03,
  1.78394147e-02,  1.78394147e-02,  1.78394147e-02,  1.78394147e-02,
  1.78394147e-02,  1.78394147e-02,  1.78394147e-02,  1.78394147e-02,
  1.78394147e-02,  1.78394147e-02,  1.78394147e-02,  1.78394147e-02,
  1.78394147e-02,  1.78394147e-02,  1.78394147e-02,  1.78394147e-02,
  1.78394147e-02,  1.78394147e-02, -1.66554297e-03,  4.35678790e-02,
  7.25381036e-02, -1.39827935e-01,  6.72862595e-02, -2.45289273e-02,
 -1.70937372e-01,  3.98806669e-02,  2.39280368e-02, -2.16571159e-03,
  3.14356490e-02, -2.98803414e-02, -1.58050341e-02,  7.77996887e-02,
  1.62535585e-02, -1.43408167e-02,  7.78675354e-03,  1.53128889e-02,
 -3.03953820e-02,  1.09800934e-02,  7.67505046e-03,  1.53577075e-02,
  2.24222113e-02, -1.10147282e-04,  3.56920591e-02, -1.34287782e-01,
  2.22385077e-02, -1.66851328e-02, -7.41986856e-02,  3.52563570e-04,
  1.57289489e-02,  1.55936569e-04, -1.66554297e-03, -1.66554297e-03,
 -1.66554297e-03, -1.66554297e-03, -1.66554297e-03, -1.66554297e-03,
 -1.66554297e-03,  1.53128889e-02,  2.77873973e-02, -5.60513651e-03,
  1.72879964e-01,  7.39478493e-02, -1.66554297e-03,  7.02757546e-03,
  7.87431437e-03,  3.95024195e-02,  2.85381895e-02,  1.72973732e-02,
  7.51412782e-02,  5.12006323e-02, -1.48438979e-02,  1.20967679e-01,
  1.39657929e-02,  4.95348386e-02,  5.83685435e-02,  4.84187368e-02,
  5.48842042e-02,  5.48842042e-02,  5.46518502e-03, -7.48982485e-02,
 -1.28998360e-01, -6.78852993e-02, -2.20895112e-01, -6.64300917e-03,
 -3.22345607e-02, -1.66554297e-03,  1.81403751e-02, -3.84950444e-02,
 -5.46184201e-03, -5.79630392e-02,  1.53128889e-02, -5.07761788e-02,
 -1.46387843e-01, -3.27181453e-02,  1.03658152e-02,  2.70607970e-02,
 -1.76488922e-02,  2.71215037e-02, -3.75069164e-02,  4.26786135e-02,
  1.28617156e-02,  3.03312252e-02,  2.09948411e-02, -8.78110015e-03,
 -7.96294413e-03,  7.80440366e-02,  7.88490031e-02,  2.52613966e-02,
  3.53508695e-02,  1.30080312e-02,  5.28950565e-02,  8.58771666e-03,
  4.28633356e-02, -4.89542215e-03, -3.20215215e-02,  3.94915159e-02,
 -5.62548212e-02,  3.51673987e-02, -1.45285842e-02, -2.04634079e-02,
  8.02015557e-02,  5.58568355e-03, -2.03885343e-02, -5.90636981e-02,
 -5.03143495e-02, -2.90759810e-02,  3.13322744e-02,  1.42354776e-02,
  3.55029354e-02,  1.54010393e-02,  3.31973956e-02,  1.67932596e-02,
  2.89030582e-02,  7.71844158e-02,  2.08670959e-02, -6.11614561e-02,
  2.85467382e-02,  1.53128889e-02,  2.35300581e-02,  9.59346467e-03,
 -5.25642350e-02,  5.85018377e-02,  4.86175938e-02,  1.37963385e-02,
 -3.16203115e-02, -4.05501066e-02,  8.26964967e-03, -7.67322097e-02,
  9.72613645e-02,  4.66453928e-02,  5.70291441e-02,  9.53020464e-02,
  7.08923040e-02,  3.17242924e-02,  3.43046886e-02, -7.66508258e-02,
 -1.41366515e-01, -2.41185666e-02,  1.23776098e-02, -1.66554297e-03,
 -1.66554297e-03, -1.66554297e-03, -1.66554297e-03, -1.66554297e-03,
  4.40964209e-02,  4.91090026e-02,  3.62924510e-02,  7.93274277e-02,
  1.15138801e-02, -5.72834577e-02, -6.40991649e-02,  4.99852884e-02,
  6.40400444e-03,  3.04501666e-02, -1.42544671e-02,  1.93020470e-02]

def run_tests_vertical_a5a():
    N_GD = 200
    y0 = np.zeros(X_train.shape[1])
    # y_GD, x_GD = gradient_descent(X_train, Y_train, N_GD, y0, L * 0.55, mu, act_val, gradf_A, f_A)
    # y_GD, x_GD = gradient_descent(X_train, Y_train, N_GD, y0, L * 0.53, mu, act_val, gradf_A, f_A)
    y_GD, x_GD = gradient_descent(X_train, Y_train, N_GD, y0, L * 0.53, mu, act_val, gradf_A, f_A)  






    y0 = np.zeros(X_train.shape[1])
    w0 = np.copy(y0)
    z0 = np.copy(y0)

    L_waved = L * 4 * (X_train.shape[0] ** 2 * L ** 2 / mu ** 2)
    sigma = mu / L_waved

    theta1 = min(np.sqrt(2 * sigma * devices_num * X_train.shape[0] / batch_size / 3), 1/2)
    stepsize = 1/2 / (1 + 1/2) / theta1

    eta_x_sigma = stepsize * sigma
    eta_div_L_waved = stepsize / L_waved
    N_katyusha_ver_permK = 1100

    # y_katyusha_ver_permK, x_katyusha_ver_permK = DVPL_Katyusha_PermK(X_train, Y_train, N_katyusha_ver_permK, devices_num, y0, w0, z0, eta_x_sigma * 1_500_000, eta_div_L_waved * 1_000_000, theta1 * 1_500_000, act_val, gradf_A, f_A, batch_size)
    # y_katyusha_ver_permK, x_katyusha_ver_permK = DVPL_Katyusha_PermK(X_train, Y_train, N_katyusha_ver_permK, devices_num, y0, w0, z0, eta_x_sigma * 1_500_000, eta_div_L_waved * 3_000_000, theta1 * 1_500_000, act_val, gradf_A, f_A, batch_size)
    y_katyusha_ver_permK, x_katyusha_ver_permK = DVPL_Katyusha_PermK(X_train, Y_train, N_katyusha_ver_permK, devices_num, y0, w0, z0, eta_x_sigma * 6_000_000, eta_div_L_waved * 3_000_000, theta1 * 1_500_000, act_val, gradf_A, f_A, batch_size)




    y0 = np.zeros(X_train.shape[1])
    w0 = np.copy(y0)
    z0 = np.copy(y0)

    L_waved = L * 4 / batch_size * (1 + (omega - 1) * X_train.shape[0] ** 2 * L ** 2 / mu ** 2)
    sigma = mu / L_waved

    theta1 = min(np.sqrt(2 * sigma * omega * X_train.shape[0] / batch_size / 3), 1/2)
    stepsize = 1/2 / (1 + 1/2) / theta1

    eta_x_sigma = stepsize * sigma
    eta_div_L_waved = stepsize / L_waved
    N_katyusha_ver_randK = 2000

    # y_katyusha_ver_randK, x_katyusha_ver_randK = DVPL_Katyusha_RandK(X_train, Y_train, N_katyusha_ver_randK, devices_num, y0, w0, z0, eta_x_sigma * 1_000_000, eta_div_L_waved * 10_000_000, theta1 * 3_000, act_val, gradf_A, f_A, RandK, K, batch_size)
    # y_katyusha_ver_randK, x_katyusha_ver_randK = DVPL_Katyusha_RandK(X_train, Y_train, N_katyusha_ver_randK, devices_num, y0, w0, z0, eta_x_sigma * 2_000_000, eta_div_L_waved * 5_000_000, theta1 * 20_000, act_val, gradf_A, f_A, RandK, K, batch_size)
    y_katyusha_ver_randK, x_katyusha_ver_randK = DVPL_Katyusha_RandK(X_train, Y_train, N_katyusha_ver_randK, devices_num, y0, w0, z0, eta_x_sigma * 1_000_000, eta_div_L_waved * 2_000_000, theta1 * 7_000, act_val, gradf_A, f_A, RandK, K, batch_size)





    omega = 1/8
    y0 = np.zeros(X_train.shape[1])
    w0 = np.copy(y0)
    z0 = np.copy(y0)

    L_waved = L * 4 / batch_size * (1 + omega * X_train.shape[0] ** 2 * L ** 2 / mu ** 2)
    sigma = mu / L_waved

    theta1 = min(np.sqrt(2 * sigma * X_train.shape[0] / 3), 1/2)
    stepsize = 1/2 / (1 + 1/2) / theta1

    eta_x_sigma = stepsize * sigma
    eta_div_L_waved = stepsize / L_waved
    N_katyusha_ver_NatDith = 1000

    # y_katyusha_ver_NatDith, x_katyusha_ver_NatDith = DVPL_Katyusha_NatDith(X_train, Y_train, N_katyusha_ver_NatDith, devices_num, y0, w0, z0, eta_x_sigma * 100_000, eta_div_L_waved * 2_000_000, theta1 * 500, act_val, gradf_A, f_A, NatDith, batch_size)
    # y_katyusha_ver_NatDith, x_katyusha_ver_NatDith = DVPL_Katyusha_NatDith(X_train, Y_train, N_katyusha_ver_NatDith, devices_num, y0, w0, z0, eta_x_sigma * 1_000, eta_div_L_waved * 1_500_000, theta1 * 500, act_val, gradf_A, f_A, NatDith, batch_size)
    y_katyusha_ver_NatDith, x_katyusha_ver_NatDith = DVPL_Katyusha_NatDith(X_train, Y_train, N_katyusha_ver_NatDith, devices_num, y0, w0, z0, eta_x_sigma * 1_0, eta_div_L_waved * 1_000_000, theta1 * 300, act_val, gradf_A, f_A, NatDith, batch_size)





    gamma = np.sqrt(L/mu)
    N_nesterov = 100

    # y_nesterov, x_nesterov = nesterov(X_train, Y_train, N_nesterov, y0, L * 1.13, gamma * 0.08, mu, act_val, gradf_A, f_A)
    # y_nesterov, x_nesterov = nesterov(X_train, Y_train, N_nesterov, y0, L * 0.8, gamma * 0.08, mu, act_val, gradf_A, f_A)
    y_nesterov, x_nesterov = nesterov(X_train, Y_train, N_nesterov, y0, L * 0.9, gamma * 0.08, mu, act_val, gradf_A, f_A)








    plt.figure(figsize=(10, 8))
    plt.title('W3A')
    plt.ylabel(R"$\log \left ( |f(x^k) - f(x^*)| \; / \; |f(x^0 - f(x^*)| \right )$")
    plt.xlabel('Transfer units No.')

    num = 7

    plt.semilogy(x_GD[np.log(y_GD) > -10], y_GD[np.log(y_GD) > -10], markersize=10, markevery= int(N_GD/num), color='orange', marker='^', label='GD')
    plt.semilogy(x_nesterov[np.log(y_nesterov) > -10], y_nesterov[np.log(y_nesterov) > -10], markersize=10, markevery=int(N_nesterov/num), color='red', marker='|', label='Nesterov')
    plt.semilogy(x_katyusha_ver_permK[np.log(y_katyusha_ver_permK) > -10], y_katyusha_ver_permK[np.log(y_katyusha_ver_permK) > -10], markersize=10, markevery=int(N_katyusha_ver_permK/num), color='blue', marker='*', label='DVPL-Katyusha+PermK')
    plt.semilogy(x_katyusha_ver_randK[np.log(y_katyusha_ver_randK) > -10], y_katyusha_ver_randK[np.log(y_katyusha_ver_randK) > -10], markersize=10, markevery=int(N_katyusha_ver_randK/num), color='green', marker='o', label='DVPL-Katyusha+RandK')
    plt.semilogy(x_katyusha_ver_NatDith[np.log(y_katyusha_ver_NatDith) > -10] / 2, y_katyusha_ver_NatDith[np.log(y_katyusha_ver_NatDith) > -10], markersize=10, markevery=int(N_katyusha_ver_NatDith/num), color='purple', marker='v', label='DVPL-Katyusha+NatDith')

    plt.legend(loc='best')

    plt.grid(True)
    plt.show()

def f_A(X, Y, w, mu):
    res = 1 / X_train.shape[0] * np.linalg.norm((np.dot(X, w) - Y), 2) ** 2 + mu / 2 * np.linalg.norm(w, 2) ** 2
    return res

def gradf_A(X, Y, w, mu):
    res = 2 / X_train.shape[0] * (np.dot(X.T, np.dot(X, w)) - np.dot(X.T, Y)) + mu * w
    return res

def RandK(vec, K):
    number_of_elements = int(K / 100 * vec.size) + 1
    chosen = np.random.choice(np.arange(vec.size), number_of_elements, replace=False)
    chosen = np.sort(chosen)
    to_be_inserted = vec[chosen]
    vec_new = np.zeros(vec.size)
    vec_new = np.insert(vec_new, chosen, to_be_inserted)
    vec_new = np.delete(vec_new, chosen + np.arange(chosen.size) + 1)
    return vec_new * 100 / K

def NatDith(vec):
    random_vec = np.random.rand(vec.size)
    exponents = np.floor(np.log2(np.abs(vec)))
    mask = random_vec < (2 ** (exponents + 1) - np.abs(vec)) / 2 ** exponents
    return np.multiply((vec > 0) * 2 - 1, 2 ** exponents * mask + 2 ** (exponents + 1) * (1 - mask))

def DVPL_Katyusha_RandK(X_train, Y_train, N, devices_num, y0, w0, z0, eta_x_sigma, eta_div_L_waved, theta1, act_val, gradf, f, compressor, K, batch_size):
    proc = np.zeros(N)
    x_axis = np.zeros(N)

    y = np.copy(y0)
    w = np.copy(w0)
    z = np.copy(z0)

    x0 = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y
    x = np.copy(x0)
    divide_by = abs(f(X_train, Y_train, x0, mu) - f(X_train, Y_train, act_val, mu))
    omega = 1 / K * 100
    p = batch_size / X_train.shape[0] / omega

    batches = np.zeros((batch_size, devices_num))
    gradf_w = gradf(X_train, Y_train, w, mu)
    datapoints_per_device = int(X_train.shape[1] / devices_num)

    for i in range(N):
        proc[i] = abs(f(X_train, Y_train, x, mu) - f(X_train, Y_train, act_val, mu)) / divide_by
        rows = np.random.choice(X_train.shape[0], batch_size, replace=False)

        x = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y

        for j in range(devices_num):
            X_deviced = []

            if (j == devices_num - 1):
                X_deviced = X_train[rows, j * datapoints_per_device :]
                batches[:, j] = np.dot(X_deviced, x[j * datapoints_per_device :] - w[j * datapoints_per_device :])
                batches[:, j] = compressor(batches[:, j], K)
            else:
                X_deviced = X_train[rows, j * datapoints_per_device : (j + 1) * datapoints_per_device]
                batches[:, j] = np.dot(X_deviced, x[j * datapoints_per_device : (j + 1) * datapoints_per_device] - w[j * datapoints_per_device : (j + 1) * datapoints_per_device])
                batches[:, j] = compressor(batches[:, j], K)

        g = np.sum(batches, axis=1)
        g = 2/batch_size * np.dot(X_train[rows].T, g) + gradf_w
        z_new = (eta_x_sigma * x + z - eta_div_L_waved * g) / (1 + eta_x_sigma)
        y = x + theta1 * (z_new - z)
        z = np.copy(z_new)

        if (random.random() < p):
            w = np.copy(y)
            gradf_w = gradf(X_train, Y_train, w, mu)
            x_axis[i] = batch_size * K / 100 + X_train.shape[0] if i == 0 else x_axis[i - 1] + batch_size * K / 100 + X_train.shape[0]
        else:
            x_axis[i] = batch_size * K / 100 if i == 0 else x_axis[i - 1] + batch_size * K / 100

    gc.collect()
    return proc, x_axis

def DVPL_Katyusha_PermK(X_train, Y_train, N, devices_num, y0, w0, z0, eta_x_sigma, eta_div_L_waved, theta1, act_val, gradf, f, batch_size):
    proc = np.zeros(N)
    x_axis = np.zeros(N)

    y = np.copy(y0)
    w = np.copy(w0)
    z = np.copy(z0)

    x0 = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y
    x = np.copy(x0)
    divide_by = abs(f(X_train, Y_train, x0, mu) - f(X_train, Y_train, act_val, mu))
    p = batch_size / X_train.shape[0] / devices_num

    batches = np.zeros((batch_size, devices_num))
    gradf_w = gradf(X_train, Y_train, w, mu)
    datapoints_per_device = int(X_train.shape[1] / devices_num)

    for i in range(N):
        proc[i] = abs(f(X_train, Y_train, x, mu) - f(X_train, Y_train, act_val, mu)) / divide_by
        rows = np.random.choice(X_train.shape[0], batch_size, replace=False)

        x = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y
        permutation = np.random.choice(batch_size, batch_size, replace=False)
        permk_per_device = int(batch_size / devices_num)

        for j in range(devices_num):
            X_deviced = []

            if (j == devices_num - 1):
                X_deviced = X_train[rows, j * datapoints_per_device :]
                batches[:, j] = np.dot(X_deviced, x[j * datapoints_per_device :] - w[j * datapoints_per_device :])
            else:
                X_deviced = X_train[rows, j * datapoints_per_device : (j + 1) * datapoints_per_device]
                batches[:, j] = np.dot(X_deviced, x[j * datapoints_per_device : (j + 1) * datapoints_per_device] - w[j * datapoints_per_device : (j + 1) * datapoints_per_device])

            this_permutation = permutation[j * permk_per_device : (j + 1) * permk_per_device]
            this_permutation = np.sort(this_permutation)
            vec_new = np.zeros(batch_size)
            vec_new = np.insert(vec_new, this_permutation, batches[this_permutation, j])
            vec_new = np.delete(vec_new, this_permutation + np.arange(this_permutation.size) + 1)
            batches[:, j] = vec_new * devices_num

        g = np.sum(batches, axis=1)
        g = 2/batch_size * np.dot(X_train[rows].T, g) + gradf_w
        z_new = (eta_x_sigma * x + z - eta_div_L_waved * g) / (1 + eta_x_sigma)
        y = x + theta1 * (z_new - z)
        z = np.copy(z_new)

        if (random.random() < p):
            w = np.copy(y)
            gradf_w = gradf(X_train, Y_train, w, mu)
            x_axis[i] = batch_size / devices_num + X_train.shape[0] if i == 0 else x_axis[i - 1] + batch_size / devices_num + X_train.shape[0]
        else:
            x_axis[i] = batch_size / devices_num if i == 0 else x_axis[i - 1] + batch_size / devices_num

    gc.collect()
    return proc, x_axis

def DVPL_Katyusha_NatDith(X_train, Y_train, N, devices_num, y0, w0, z0, eta_x_sigma, eta_div_L_waved, theta1, act_val, gradf, f, compressor, batch_size):
    proc = np.zeros(N)
    x_axis = np.zeros(N)

    y = np.copy(y0)
    w = np.copy(w0)
    z = np.copy(z0)

    x0 = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y
    x = np.copy(x0)
    divide_by = abs(f(X_train, Y_train, x0, mu) - f(X_train, Y_train, act_val, mu))
    omega = 1/8
    p = batch_size / X_train.shape[0] / omega

    batches = np.zeros((batch_size, devices_num))
    gradf_w = gradf(X_train, Y_train, w, mu)
    datapoints_per_device = int(X_train.shape[1] / devices_num)

    for i in range(N):
        proc[i] = abs(f(X_train, Y_train, x, mu) - f(X_train, Y_train, act_val, mu)) / divide_by
        rows = np.random.choice(X_train.shape[0], batch_size, replace=False)

        x = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y

        for j in range(devices_num):
            X_deviced = []

            if (j == devices_num - 1):
                X_deviced = X_train[rows, j * datapoints_per_device :]
                batches[:, j] = np.dot(X_deviced, x[j * datapoints_per_device :] - w[j * datapoints_per_device :])
                batches[:, j] = compressor(batches[:, j])
            else:
                X_deviced = X_train[rows, j * datapoints_per_device : (j + 1) * datapoints_per_device]
                batches[:, j] = np.dot(X_deviced, x[j * datapoints_per_device : (j + 1) * datapoints_per_device] - w[j * datapoints_per_device : (j + 1) * datapoints_per_device])
                batches[:, j] = compressor(batches[:, j])

        g = np.sum(batches, axis=1)
        g = 2/batch_size * np.dot(X_train[rows].T, g) + gradf_w
        z_new = (eta_x_sigma * x + z - eta_div_L_waved * g) / (1 + eta_x_sigma)
        y = x + theta1 * (z_new - z)
        z = np.copy(z_new)

        if (random.random() < p):
            w = np.copy(y)
            gradf_w = gradf(X_train, Y_train, w, mu)
            x_axis[i] = batch_size * 9/64 + X_train.shape[0] if i == 0 else x_axis[i - 1] + batch_size * 9/64 + X_train.shape[0]
        else:
            x_axis[i] = batch_size * 9/64 if i == 0 else x_axis[i - 1] + batch_size * 9/64

    gc.collect()
    return proc, x_axis

def gradient_descent(X_train, Y_train, N, x0, L, mu, act_val, gradf, f):
    x = np.copy(x0)

    proc = np.zeros(N)
    x_axis = np.zeros(N)

    divide_by = abs(f(X_train, Y_train, x0, mu) - f(X_train, Y_train, act_val, mu))

    for i in range(N):
        proc[i] = abs(f(X_train, Y_train, x, mu) - f(X_train, Y_train, act_val, mu)) / divide_by
        x = x - 1 / L * gradf(X_train, Y_train, x, mu)
        x_axis[i] = X_train.shape[0] if i == 0 else x_axis[i - 1] + X_train.shape[0]

    return proc, x_axis

def nesterov(X_train, Y_train, N, x0, L, gamma, mu, act_val, gradf, f):
    x_old = np.copy(x0)
    x_new = np.copy(x0)

    proc = np.zeros(N)
    x_axis = np.zeros(N)

    divide_by = abs(f(X_train, Y_train, x0, mu) - f(X_train, Y_train, act_val, mu))

    for i in range(N):
        proc[i] = abs(f(X_train, Y_train, x_new, mu) - f(X_train, Y_train, act_val, mu)) / divide_by

        y = x_new + gamma * (x_new - x_old)
        x_old = np.copy(x_new)
        x_new = y - 1 / L * gradf(X_train, Y_train, y, mu)

        x_axis[i] = X_train.shape[0] if i == 0 else x_axis[i - 1] + X_train.shape[0]

    gc.collect()
    return proc, x_axis