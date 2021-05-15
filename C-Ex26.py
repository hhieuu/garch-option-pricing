# Sample solution C26, WT2020

import math
import numpy as np
from scipy import optimize


# part a)
# from previous exercise
def sigma_sq_historic(theta, x):
    # number of observations
    n = len(x)

    # recursion for sigma parameters
    sigma_sq = np.ones(n)
    sigma_sq[0] = theta[3] ** 2
    for i in range(1, n):
        sigma_sq[i] = theta[0] + theta[1] * x[i - 1] ** 2 + theta[2] * sigma_sq[i - 1]
    return sigma_sq


def log_likelihood_GARCH_11(theta, x):
    # number of observations
    n = len(x)

    # recursion for sigma parameters
    sigma_sq = sigma_sq_historic(theta, x)

    # log likelihood
    y = -0.5 * (n * math.log(2 * math.pi) + np.sum(np.log(sigma_sq)))
    y -= 0.5 * np.sum(np.divide(np.power(x, 2), sigma_sq))

    return y


# for the optimization we need the negative log likelihood function
def neg_log_likelihood_GARCH_11(theta, x):
    y = log_likelihood_GARCH_11(theta, x)
    return -y


# part b)
def estimates_GARCH_11(x_data):
    # initial theta parameters
    theta_sigma1_0 = np.array([0.01, 0.2, 0.8, np.std(x_data)])

    # optimization command
    theta_sigma1_hat = optimize.minimize(neg_log_likelihood_GARCH_11, args=x_data, x0=theta_sigma1_0,
                                         bounds=((0.0001, 5), (0.00001, 5), (0.00001, 5), (0.001, 5)))

    return theta_sigma1_hat.x


# part c)
# from previous exercise
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


def VaR_ES_GARCH11_MC(k, m, l, alpha, x):
    # estimate theta
    theta_hat = estimates_GARCH_11(x)

    # compute sigma for the next trading day's risk factor change
    sigma_sq_hat = sigma_sq_historic(theta_hat, x)
    sigma1_new = math.sqrt(theta_hat[0] + theta_hat[1] * x[-1] ** 2 + theta_hat[2] * sigma_sq_hat[-1])

    # Application of GARCH_11_MC
    X = GARCH11_MC(k, m, theta_hat[0:3], sigma1_new)[0]

    # losses
    loss = np.ones(m)
    for j in range(0, m):
        loss[j] = l(X[j, :])

    # Value at Risk and Expected Shortfall
    l_data_sorted = np.flip(np.sort(loss))

    VaR = l_data_sorted[int(np.floor(m * (1 - alpha)) + 1)]
    ES = 1 / (np.floor(m * (1 - alpha)) + 1) * np.sum(l_data_sorted[0:int(np.floor(m * (1 - alpha)) + 1)])
    return VaR, ES


# part d)

# application to DAX time series
# level Value at Risk
alpha = 0.98

# number of simulations in Monte-Carlo method
m = 1000

# forecast period
k = 5

# load DAX time series
dax = np.flip(np.genfromtxt('.\Data\dax_data.csv', delimiter=';', skip_header=1, usecols=4))
# compute log returns
x = np.diff(np.log(dax))


# loss operator
def l(x):
    return dax[-1] * (1 - np.exp(np.sum(x)))


# parameter estimation
theta_hat = estimates_GARCH_11(x)
print('Estimated parameters for theta and sigma1: ' + str(theta_hat))

# compute VaR and ES
VaR, ES = VaR_ES_GARCH11_MC(k, m, l, alpha, x)

# display the result
print(str(k) + '-day ahead MC estimates:')
print('    VaR: ' + str(VaR) + '          Es: ' + str(ES))
