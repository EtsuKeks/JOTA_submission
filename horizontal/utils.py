import numpy as np
import random

L, mu, devices_num, w0, X, Y, act_val = None, None, None, None, None, None, None

def init(L_value, mu_value, devices_num_value, w0_value, X_value, Y_value, act_val_value):
    global L, mu, devices_num, w0, X, Y, act_val
    L = L_value
    mu = mu_value
    devices_num = devices_num_value
    w0 = w0_value
    X = X_value
    Y = Y_value
    act_val = act_val_value

def f(X, Y, w, mu):
    ywx = -np.multiply(np.dot(X, w), Y)
    to_be_summed = np.log(1 + np.exp(ywx))
    res = 1 / X.shape[0] * np.sum(to_be_summed)
    res += np.linalg.norm(w, 2) ** 2 * mu / 2
    return res

def gradf(X, Y, w, mu, X_diag_y):
    ywx = -np.multiply(np.dot(X, w), Y)
    to_be_multiplied = np.exp(ywx) / (1 + np.exp(ywx))
    res = 1 / X.shape[0] * np.dot(-X_diag_y, to_be_multiplied)
    res += mu * w
    return res

def Rand1divn(vec):
    global L, mu, devices_num, w0, X, Y, act_val

    K = 1 / devices_num
    number_of_components = int(K * vec.size) + 1
    chosen = np.random.choice(np.arange(vec.size), number_of_components, replace=False)
    chosen = np.sort(chosen)
    to_be_inserted = vec[chosen]
    vec_new = np.zeros(vec.size)
    vec_new = np.insert(vec_new, chosen, to_be_inserted)
    vec_new = np.delete(vec_new, chosen + np.arange(chosen.size) + 1)
    return vec_new / K

def GD(N, L_times):
    global L, mu, devices_num, w0, X, Y, act_val

    stepsize = 1 / (L * L_times)

    x = np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by, X_diag_y = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu)), np.dot(X.T, np.diag(Y))

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x = x - stepsize * gradf(X, Y, x, mu, X_diag_y)

        x_axis[i] = X.shape[1] if i == 0 else x_axis[i - 1] + X.shape[1]

    return x_axis, y_axis

def AGD(N):
    global L, mu, devices_num, w0, X, Y, act_val

    gamma = np.sqrt(L / mu)
    theta = 1 / (1 + 1 / gamma)
    stepsize = 1 / L

    x, x_g, x_f_new, x_f_old = np.copy(w0), np.copy(w0), np.copy(w0), np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by, X_diag_y = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu)), np.dot(X.T, np.diag(Y))

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x_g = theta * x_f_old + (1 - theta) * x
        x_f_new = x_g - stepsize * gradf(X, Y, x_g, mu, X_diag_y)
        x = gamma * (x_f_new - x_f_old) + x_f_old
        x_f_old = np.copy(x_f_new)

        x_axis[i] = X.shape[1] if i == 0 else x_axis[i - 1] + X.shape[1]

    return x_axis, y_axis

def ADIANA(N, eta_times):
    global L, mu, devices_num, w0, X, Y, act_val

    k = L / mu
    omega = devices_num
    theta1 = 1 / (3 * np.sqrt(k))
    theta2 = 1 / (3 * np.sqrt(devices_num) + 3 * devices_num / omega)
    eta = devices_num * theta2 / 120 / omega / L * eta_times
    p = 1 / (1 + omega)
    alpha = 1 / (1 + omega)
    beta = 2 * theta1 / (2 * theta1 + eta * mu)
    gamma = eta / (2 * theta1 + eta * mu)

    x, y, w, z = np.copy(w0), np.copy(w0), np.copy(w0), np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu))

    m_s, c_s, h_s = np.zeros((w0.size, devices_num)), np.zeros((w0.size, devices_num)), np.zeros((w0.size, devices_num))
    g, h = np.zeros(w0.size), np.zeros(w0.size)
    datapoints_per_device = int(X.shape[0] / devices_num)

    X_deviced_s, Y_deviced_s, X_diag_y_s = [], [], []
    for j in range(devices_num):
        X_deviced = []
        Y_deviced = []

        if (j == devices_num - 1):
            X_deviced = X[j * datapoints_per_device :, :]
            Y_deviced = Y[j * datapoints_per_device :]
        else:
            X_deviced = X[j * datapoints_per_device : (j + 1) * datapoints_per_device, :]
            Y_deviced = Y[j * datapoints_per_device : (j + 1) * datapoints_per_device]

        X_deviced_s.append(X_deviced)
        Y_deviced_s.append(Y_deviced)
        X_diag_y_s.append(np.dot(X_deviced.T, np.diag(Y_deviced)))

    for i in range(N):
        y_axis[i] = abs(f(X, Y, y, mu) - f(X, Y, act_val, mu)) / divide_by

        x = theta1 * z + theta2 * w + (1 - theta1 - theta2) * y

        for j in range(devices_num):
            m_s[:, j] = Rand1divn(gradf(X_deviced_s[j], Y_deviced_s[j], x, mu, X_diag_y_s[j]) - h_s[:, j])
            c_s[:, j] = Rand1divn(gradf(X_deviced_s[j], Y_deviced_s[j], w, mu, X_diag_y_s[j]) - h_s[:, j])
            h_s[:, j] = h_s[:, j] + alpha * c_s[:, j]

        g = h + 1 / devices_num * np.sum(m_s, axis=1)
        h = h + alpha / devices_num * np.sum(c_s, axis=1)
        y = x - eta * g
        z = beta * z + (1 - beta) * x + gamma / eta * (y - x)

        if (random.random() < p):
            w = np.copy(y)

        x_axis[i] = 2 * X.shape[1] / devices_num if i == 0 else x_axis[i - 1] + 2 * X.shape[1] / devices_num

    return x_axis, y_axis

def DHPL_Katyusha_Rand1divn(N):
    global L, mu, devices_num, w0, X, Y, act_val

    omega = devices_num
    L_waved = L * max(1, omega / devices_num)
    sigma = mu / L_waved
    theta1 = min(np.sqrt(2 * sigma * omega / 3), 1/2)
    eta = 1/2 / (1 + 1/2) / theta1
    p = 1 / omega
    eta_x_sigma = eta * sigma
    eta_div_L_waved = eta / L_waved

    x, y, w, z = np.copy(w0), np.copy(w0), np.copy(w0), np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by, X_diag_y = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu)), np.dot(X.T, np.diag(Y))

    gs = np.zeros((w0.size, devices_num))
    gradf_w = gradf(X, Y, w, mu, X_diag_y)
    datapoints_per_device = int(X.shape[0] / devices_num)

    X_deviced_s, Y_deviced_s, X_diag_y_s = [], [], []
    for j in range(devices_num):
        X_deviced = []
        Y_deviced = []

        if (j == devices_num - 1):
            X_deviced = X[j * datapoints_per_device :, :]
            Y_deviced = Y[j * datapoints_per_device :]
        else:
            X_deviced = X[j * datapoints_per_device : (j + 1) * datapoints_per_device, :]
            Y_deviced = Y[j * datapoints_per_device : (j + 1) * datapoints_per_device]

        X_deviced_s.append(X_deviced)
        Y_deviced_s.append(Y_deviced)
        X_diag_y_s.append(np.dot(X_deviced.T, np.diag(Y_deviced)))

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y

        for j in range(devices_num):
            gs[:, j] = gradf(X_deviced_s[j], Y_deviced_s[j], x, mu, X_diag_y_s[j]) - gradf(X_deviced_s[j], Y_deviced_s[j], w, mu, X_diag_y_s[j])
            gs[:, j] = Rand1divn(gs[:, j])

        g = 1 / devices_num * np.sum(gs, axis=1) + gradf_w
        z_new = (eta_x_sigma * x + z - eta_div_L_waved * g) / (1 + eta_x_sigma)
        y = x + theta1 * (z_new - z)
        z = np.copy(z_new)

        if (random.random() < p):
            w = np.copy(y)
            gradf_w = gradf(X, Y, w, mu, X_diag_y)
            x_axis[i] = X.shape[1] / devices_num + X.shape[1] if i == 0 else x_axis[i - 1] + X.shape[1] / devices_num + X.shape[1]
        else:
            x_axis[i] = X.shape[1] / devices_num if i == 0 else x_axis[i - 1] + X.shape[1] / devices_num

    return x_axis, y_axis

def DHPL_Katyusha_PermK(N):
    global L, mu, devices_num, w0, X, Y, act_val

    L_waved = L
    sigma = mu / L_waved
    theta1 = min(np.sqrt(2 * sigma * devices_num / 3), 1/2)
    eta = 1/2 / (1 + 1/2) / theta1
    p = 1 / devices_num
    eta_x_sigma = eta * sigma
    eta_div_L_waved = eta / L_waved

    x, y, w, z = np.copy(w0), np.copy(w0), np.copy(w0), np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by, X_diag_y = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu)), np.dot(X.T, np.diag(Y))

    gs = np.zeros((w0.size, devices_num))
    gradf_w = gradf(X, Y, w, mu, X_diag_y)
    datapoints_per_device = int(X.shape[0] / devices_num)

    X_deviced_s, Y_deviced_s, X_diag_y_s = [], [], []
    for j in range(devices_num):
        X_deviced = []
        Y_deviced = []

        if (j == devices_num - 1):
            X_deviced = X[j * datapoints_per_device :, :]
            Y_deviced = Y[j * datapoints_per_device :]
        else:
            X_deviced = X[j * datapoints_per_device : (j + 1) * datapoints_per_device, :]
            Y_deviced = Y[j * datapoints_per_device : (j + 1) * datapoints_per_device]

        X_deviced_s.append(X_deviced)
        Y_deviced_s.append(Y_deviced)
        X_diag_y_s.append(np.dot(X_deviced.T, np.diag(Y_deviced)))

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y
        permutation = np.random.choice(X.shape[1], X.shape[1], replace=False)
        permk_per_device = int(X.shape[1] / devices_num)

        for j in range(devices_num):
            gs[:, j] = gradf(X_deviced_s[j], Y_deviced_s[j], x, mu, X_diag_y_s[j]) - gradf(X_deviced_s[j], Y_deviced_s[j], w, mu, X_diag_y_s[j])

            if (j == devices_num - 1):
                this_permutation = permutation[j * permk_per_device :]
            else:
                this_permutation = permutation[j * permk_per_device : (j + 1) * permk_per_device]

            this_permutation = np.sort(this_permutation)
            vec_new = np.zeros(X.shape[1])
            vec_new = np.insert(vec_new, this_permutation, gs[this_permutation, j])
            vec_new = np.delete(vec_new, this_permutation + np.arange(this_permutation.size) + 1)
            gs[:, j] = vec_new * devices_num

        g = 1 / devices_num * np.sum(gs, axis=1) + gradf_w
        z_new = (eta_x_sigma * x + z - eta_div_L_waved * g) / (1 + eta_x_sigma)
        y = x + theta1 * (z_new - z)
        z = np.copy(z_new)

        if (random.random() < p):
            w = np.copy(y)
            gradf_w = gradf(X, Y, w, mu, X_diag_y)
            x_axis[i] = X.shape[1] / devices_num + X.shape[1] if i == 0 else x_axis[i - 1] + X.shape[1] / devices_num + X.shape[1]
        else:
            x_axis[i] = X.shape[1] / devices_num if i == 0 else x_axis[i - 1] + X.shape[1] / devices_num

    return x_axis, y_axis