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
    res = 1 / X.shape[0] * np.linalg.norm((np.dot(X, w) - Y), 2) ** 2 + mu / 2 * np.linalg.norm(w, 2) ** 2
    return res

def gradf(X, Y, w, mu):
    res = 2 / X.shape[0] * (np.dot(X.T, np.dot(X, w)) - np.dot(X.T, Y)) + mu * w
    return res

def RandK(vec, K_times):
    number_of_elements = int(K_times * vec.size) + 1
    chosen = np.random.choice(vec.size, number_of_elements, replace=False)
    chosen = np.sort(chosen)
    to_be_inserted = vec[chosen]
    vec_new = np.zeros(vec.size)
    vec_new = np.insert(vec_new, chosen, to_be_inserted)
    vec_new = np.delete(vec_new, chosen + np.arange(chosen.size) + 1)
    return vec_new / K_times

def GD(N, L_times):
    global L, mu, devices_num, w0, X, Y, act_val
    s, d = X.shape[0], X.shape[1]

    stepsize = 1 / (L * L_times)

    x = np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu))

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x = x - stepsize * gradf(X, Y, x, mu)

        x_axis[i] = s if i == 0 else x_axis[i - 1] + s

    return x_axis, y_axis

def AGD(N):
    global L, mu, devices_num, w0, X, Y, act_val
    s, d = X.shape[0], X.shape[1]

    gamma = np.sqrt(L / mu)
    theta = 1 / (1 + 1 / gamma)
    stepsize = 1 / L

    x, x_g, x_f_new, x_f_old = np.copy(w0), np.copy(w0), np.copy(w0), np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu))

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x_g = theta * x_f_old + (1 - theta) * x
        x_f_new = x_g - stepsize * gradf(X, Y, x_g, mu)
        x = gamma * (x_f_new - x_f_old) + x_f_old
        x_f_old = np.copy(x_f_new)

        x_axis[i] = s if i == 0 else x_axis[i - 1] + s

    return x_axis, y_axis

def DVPL_Katyusha_RandKAlg2(N, K_times):
    global L, mu, devices_num, w0, X, Y, act_val
    s, d = X.shape[0], X.shape[1]

    L_j_s = np.zeros(s)
    for i in range(s):
        L_j_s[i] = np.abs(np.max(np.linalg.eig(2 * np.outer(X[i, :], X[i, :]) + mu * np.eye(d))[0]))
    L_hat = np.sum(L_j_s) / s

    K = int(s * K_times)

    probas = L_j_s / s / L_hat
    L_waved = max(L, L_hat / K)
    sigma = mu / L_waved
    theta1 = min(np.sqrt(2 * sigma * s / 3 / K), 1/2)
    eta = 1/2 / (1 + 1/2) / theta1
    p = K / s
    eta_x_sigma = eta * sigma
    eta_div_L_waved = eta / L_waved

    x, y, w, z = np.copy(w0), np.copy(w0), np.copy(w0), np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu))

    gradf_w = gradf(X, Y, w, mu)

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y
        rows = np.random.choice(s, K, replace=True, p=probas)
        X_batched = X[rows]
        Y_batched = Y[rows]

        delta_pos = np.zeros(d)
        delta_neg = np.zeros(d)
        for j in range(K):
            delta_pos += (2 * (np.dot(X_batched[j, :].T, np.dot(X_batched[j, :], x)) - np.dot(X_batched[j, :].T, Y_batched[j])) + mu * x) / probas[rows[j]]
            delta_neg += (2 * (np.dot(X_batched[j, :].T, np.dot(X_batched[j, :], w)) - np.dot(X_batched[j, :].T, Y_batched[j])) + mu * w) / probas[rows[j]]

        g = gradf_w + (delta_pos - delta_neg) / K / s
        z_new = (eta_x_sigma * x + z - eta_div_L_waved * g) / (1 + eta_x_sigma)
        y = x + theta1 * (z_new - z)
        z = np.copy(z_new)

        if (random.random() < p):
            w = np.copy(y)
            gradf_w = gradf(X, Y, w, mu)
            x_axis[i] = K + s if i == 0 else x_axis[i - 1] + K + s
        else:
            x_axis[i] = K if i == 0 else x_axis[i - 1] + K

    return x_axis, y_axis

def DVPL_Katyusha_RandKAlg3(N, K_times, L_waved_times):
    global L, mu, devices_num, w0, X, Y, act_val
    s, d = X.shape[0], X.shape[1]

    K = int(s * K_times)

    L_waved = L * L_waved_times
    sigma = mu / L_waved
    theta1 = min(np.sqrt(2 * sigma * s / 3 / K), 1/2)
    eta = 1/2 / (1 + 1/2) / theta1
    p = K / s
    eta_x_sigma = eta * sigma
    eta_div_L_waved = eta / L_waved

    x, y, w, z = np.copy(w0), np.copy(w0), np.copy(w0), np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu))

    linear_models = np.zeros((s, devices_num))
    gradf_w = gradf(X, Y, w, mu)
    datapoints_per_device = int(d / devices_num)

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y
        for j in range(devices_num):
            X_deviced = []

            if (j == devices_num - 1):
                X_deviced = X[:, j * datapoints_per_device :]
                linear_models[:, j] = np.dot(X_deviced, \
                                             x[j * datapoints_per_device :] - \
                                             w[j * datapoints_per_device :])
                linear_models[:, j] = RandK(linear_models[:, j], K_times)
            else:
                X_deviced = X[:, j * datapoints_per_device : (j + 1) * datapoints_per_device]
                linear_models[:, j] = np.dot(X_deviced, \
                                             x[j * datapoints_per_device : (j + 1) * datapoints_per_device] - \
                                             w[j * datapoints_per_device : (j + 1) * datapoints_per_device])
                linear_models[:, j] = RandK(linear_models[:, j], K_times)

        g = 2 / s * np.dot(X.T, np.sum(linear_models, axis=1)) + gradf_w
        z_new = (eta_x_sigma * x + z - eta_div_L_waved * g) / (1 + eta_x_sigma)
        y = x + theta1 * (z_new - z)
        z = np.copy(z_new)

        if (random.random() < p):
            w = np.copy(y)
            gradf_w = gradf(X, Y, w, mu)
            x_axis[i] = K + s if i == 0 else x_axis[i - 1] + K + s
        else:
            x_axis[i] = K if i == 0 else x_axis[i - 1] + K

    return x_axis, y_axis

def DVPL_Katyusha_PermK(N, L_waved_times):
    global L, mu, devices_num, w0, X, Y, act_val
    s, d = X.shape[0], X.shape[1]

    K = int(s / devices_num)

    L_waved = L * L_waved_times
    sigma = mu / L_waved
    theta1 = min(np.sqrt(2 * sigma * s / 3 / K), 1/2)
    eta = 1/2 / (1 + 1/2) / theta1
    p = K / s
    eta_x_sigma = eta * sigma
    eta_div_L_waved = eta / L_waved

    x, y, w, z = np.copy(w0), np.copy(w0), np.copy(w0), np.copy(w0)
    x_axis, y_axis = np.zeros(N), np.zeros(N)
    divide_by = abs(f(X, Y, w0, mu) - f(X, Y, act_val, mu))

    linear_models = np.zeros((s, devices_num))
    gradf_w = gradf(X, Y, w, mu)
    datapoints_per_device = int(d / devices_num)

    for i in range(N):
        y_axis[i] = abs(f(X, Y, x, mu) - f(X, Y, act_val, mu)) / divide_by

        x = theta1 * z + 1/2 * w + (1 - theta1 - 1/2) * y

        permutation = np.random.choice(s, s, replace=False)
        permk_per_device = int(s / devices_num) + 1
        for j in range(devices_num):
            X_deviced = []

            if (j == devices_num - 1):
                X_deviced = X[:, j * datapoints_per_device :]
                linear_models[:, j] = np.dot(X_deviced, \
                                             x[j * datapoints_per_device :] - \
                                             w[j * datapoints_per_device :])
                permutation_deviced = permutation[j * permk_per_device :]
            else:
                X_deviced = X[:, j * datapoints_per_device : (j + 1) * datapoints_per_device]
                linear_models[:, j] = np.dot(X_deviced, \
                                             x[j * datapoints_per_device : (j + 1) * datapoints_per_device] - \
                                             w[j * datapoints_per_device : (j + 1) * datapoints_per_device])
                permutation_deviced = permutation[j * permk_per_device : (j + 1) * permk_per_device]

            permutation_deviced = np.sort(permutation_deviced)
            vec_new = np.zeros(s)
            vec_new = np.insert(vec_new, permutation_deviced, linear_models[permutation_deviced, j])
            vec_new = np.delete(vec_new, permutation_deviced + np.arange(permutation_deviced.size) + 1)
            linear_models[:, j] = vec_new * devices_num

        g = 2 / s * np.dot(X.T, np.sum(linear_models, axis=1)) + gradf_w
        z_new = (eta_x_sigma * x + z - eta_div_L_waved * g) / (1 + eta_x_sigma)
        y = x + theta1 * (z_new - z)
        z = np.copy(z_new)

        if (random.random() < p):
            w = np.copy(y)
            gradf_w = gradf(X, Y, w, mu)
            x_axis[i] =  K + s if i == 0 else x_axis[i - 1] + K + s
        else:
            x_axis[i] = K if i == 0 else x_axis[i - 1] + K

    return x_axis, y_axis