# Sample solution C15, WT2020

import numpy as np


def GARCH11_MC(n, m, theta, sigma1):
    # simulation for X: m is the number of paths, n the length of the paths
    X = np.zeros((m, n))
    sigma_sq = np.zeros((m, n))

    # initial value of sigma is known
    sigma_sq[:, 0] = sigma1 ** 2

    # simulation of the required N(0,1)-distributed random variables
    Y = np.random.normal(size=(m, n))

    # initial values of X
    X[:, 0] = sigma1 * Y[:, 0]

    # recursion for X at times i>=2
    for i in range(1, n):
        sigma_sq[:, i] = theta[0] + theta[1] * np.power(X[:, i - 1], 2) + theta[2] * sigma_sq[:, i - 1]
        X[:, i] = np.multiply(np.sqrt(sigma_sq[:, i]), Y[:, i])

    return X, sigma_sq


def VaR_ES_GARCH11_MC(k, m, l, alpha, theta, sigma1):
    # Application of GARCH_11_MC
    X = GARCH11_MC(k, m, theta, sigma1)[0]

    # losses
    loss = np.ones(m)
    for j in range(0, m):
        loss[j] = l(X[j, :])

    # Value at Risk and Expected Shortfall
    l_data_sorted = np.flip(np.sort(loss))

    VaR = l_data_sorted[int(np.floor(m * (1 - alpha)) + 1)]
    ES = 1 / (np.floor(m * (1 - alpha)) + 1) * np.sum(l_data_sorted[0:int(np.floor(m * (1 - alpha)) + 1)])
    return VaR, ES


# level Value at Risk
alpha = 0.975

# number of simulations in Monte-Carlo method
m = 200

# forecast period
k = 10

# load DAX time series
dax = np.flip(np.genfromtxt('dax_data.csv', delimiter=';', skip_header=1, usecols=4))
# compute log returns
x = np.diff(np.log(dax))


# loss operator
def l(x):
    return dax[-1] * (1 - np.exp(np.sum(x)))


# given parameters for GARCH(1,1)
theta = np.array([0.0000032, 0.08426, 0.8986])
sigma1 = 0.0195

# compute sigma for the next trading day's risk factor change
n = len(x)

# recursion for sigma parameters
sigma_sq = np.ones(n+1)
sigma_sq[0] = sigma1 ** 2

for i in range(1, n+1):
    sigma_sq[i] = theta[0] + theta[1] * x[i - 1] ** 2 + theta[2] * sigma_sq[i - 1]

# compute VaR and ES
VaR, ES = VaR_ES_GARCH11_MC(k, m, l, alpha, theta, np.sqrt(sigma_sq[-1]))

# display the results
print(str(k) + '-day ahead MC estimates:')
print('VaR: ' + str(VaR) + '      ES: ' + str(ES))
