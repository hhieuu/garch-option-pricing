import math
import time

from numba import jit
import numpy as np
import pandas as pd
from scipy import optimize


@jit(nopython=True)
def compute_xi_t(return_t, risk_free_rate, sigma_t):
    """
    Compute innovation xi at time t as a function of return t, rf rate, and sigma t
    """
    return return_t - risk_free_rate + 1/2 * sigma_t


@jit(nopython=True)
def compute_log_return(price_array: np.ndarray):
    """
    Compute log return: $\ln \frac{X_t}{X_{t-1}}
    """
    return np.log(price_array[1:] / price_array[:-1])


@jit(nopython=True)
def compute_sigma_at_t(theta, lamb, sigma_sq_prev, xi_prev):
    """
    Calculate conditional variance at time t + 1 given parameters of time t
    
    Param
    -----
    theta: Tuple(Float, Float, Float, Float, Float)
        GARCH(1, 1) parameter: (alpha_0, alpha_1, beta_1, risk premium lambda, standard deviation sigma (not used))
    
    sigma_prev: Float
        Conditional variance at t - 1
            
    xi_prev: Float
        Innovation at t - 1
            
    Return
    ------
    Float:
        Conditional variance at time
    """
    a0, a1, b1, sigma0 = theta
    sigma_t = a0 + a1 * (xi_prev - lamb * math.sqrt(sigma_sq_prev)) ** 2 + b1 * sigma_sq_prev
    return sigma_t


@jit(nopython=True)
def compute_sigma_sq_historic(theta, lamb, risk_free_rate, return_array):
    """
    Calculate historical conditional variances at each epsilon
    
    Param
    -----
    theta: Tuple(Float, Float, Float, Float, Float)
        GARCH(1, 1) parameter: (alpha_0, alpha_1, beta_1, risk premium lambda, standard deviation sigma)
               
    risk_free_rate: float
        risk free rate
        
    s_array: ndarray
        Array of historical prices
        
    Return
    ------
    ndarray
        Array of historical conditional variances
    """
    # params
    n = len(return_array)
    a0, a1, b1, sigma0 = theta
    
    # init values
    # conditional variance sigma_square
    sigma_sq_array = np.ones(n)
    sigma_sq_array[0] = sigma0 ** 2
    # innovation xi
    xi_array = np.ones(n)
    xi_array[0] = compute_xi_t(return_array[0], risk_free_rate, sigma_sq_array[0]) # initial xi_0
    
    # recursion for sigma parameters
    for t in range(1, n):
        sigma_sq_prev = sigma_sq_array[t - 1]
        xi_prev = xi_array[t - 1]
        sigma_sq_t = compute_sigma_at_t(theta, lamb, sigma_sq_prev, xi_prev)
        xi_array_t = compute_xi_t(return_array[t], risk_free_rate, np.sqrt(sigma_sq_t))
        sigma_sq_array[t] = sigma_sq_t
        xi_array[t] = xi_array_t
        
    return sigma_sq_array, xi_array


@jit(nopython=True)
def neg_log_likelihood_GARCH_11(theta, lamb, risk_free_rate, return_array):
    """
    Calculate Log Likelihood of a GARCH(1, 1) process
    
    Param
    -----
    theta: Tuple(Float, Float, Float, Float, Float)
        GARCH(1, 1) parameter: (alpha_0, alpha_1, beta_1, risk premium lambda, standard deviation sigma)
        
    x: ndarray
    
    Return
    ------
    Float
        Negative log-likelihood of GARCH(1, 1 function)
        
    """
    # number of observations
    n = len(return_array)

    # recursion for sigma parameters
    sigma_sq_array, _ = compute_sigma_sq_historic(theta, lamb, risk_free_rate, return_array)
    # log likelihood
    y = -0.5 * (n * np.log(2.0 * np.pi) + np.sum(np.log(sigma_sq_array)))
    y -= 0.5 * np.sum(return_array ** 2 / sigma_sq_array)
    return - y


def estimate_GARCH_11_theta(return_array, lamb, risk_free_rate):
    """
    Estimates parameters of GARCH(1, 1) by Maximum Likelihood Estimation and Scipy's minimization implementation
    
    Param
    -----
    return_array: ndarray
        Array of returns
        
    risk_free_rate: Float
        Given return rate of risk-free asset
        
    Return
    ------
    Tuple(Float, Float, Float, Float, Float)
        Estimation result for given data and risk free rate.
        Elements are: 
            - alpha_0: bias term, 
            - alpha_1: coefficient of heteroskedastic component,
            - beta_1: coefficient of auto-regressive component,
            - lambda: risk premium
            - sigma_0: initial standard deviation (s.d. at time 0)
    """
    # initial theta parameters
    theta_sigma1_0 = np.array([0.1, 0.2, 0.8, np.std(return_array)])

    # counter
    it_counter = IterationCounter(lamb, risk_free_rate, return_array)
    
    # optimization command
    theta_opt = optimize.minimize(fun=neg_log_likelihood_GARCH_11, 
                                  args=(lamb, risk_free_rate, return_array),
                                  x0=theta_sigma1_0,
                                  bounds=((0.0001, 5), (0.00001, 5), (0.00001, 5), (0.001, 5)),
                                  method='L-BFGS-B',
                                  callback=it_counter)
    print(f"Optimization status: {theta_opt.success} after {theta_opt.nit} iterations")
    print(f"Optimization message: {theta_opt.status} - {theta_opt.message}")
    return theta_opt.x


class IterationCounter:
    """
    Class to monitor optimization process by calculating and logging out 
    """
    def __init__(self, lamb, risk_free_rate, return_array):
        self.num_it = 0
        # time
        self.init_time = time.time()
        self.last_it_time = None
        self.current_it_duration = None
        
        # opt args
        self.risk_premium = lamb
        self.risk_free_rate = risk_free_rate
        self.return_array = return_array
        
        print("Start recording iterations' status")
        
    def __call__(self, *args, **kwargs):
        # time
        current_time = time.time()
        self.num_it += 1
        if self.last_it_time is None:
            self.current_it_duration = current_time - self.init_time
        else:
            self.current_it_duration = current_time - self.last_it_time
        
        self.last_it_time = current_time
        
        # current optimization value
        current_theta = args[0]
        current_neg_llh = neg_log_likelihood_GARCH_11(current_theta, self.risk_premium, self.risk_free_rate, self.return_array)
        
        print('==========================================')
        print(f"Number of iterations      : {self.num_it}")
        print(f"Total elapsed time        : {current_time - self.init_time:.4f} seconds")
        print(f"Last iteration's duration : {self.current_it_duration:.4f} seconds")
        print(f"Current neg llh value     : {current_neg_llh:.6f}")
        print(f"-- alpha_0                : {current_theta[0]:.6f}")
        print(f"-- alpha_1                : {current_theta[1]:.6f}")
        print(f"-- beta_1                 : {current_theta[2]:.6f}")
        # print(f"-- lambda                 : {current_theta[3]:.6f}")
        print(f"-- init_sigma             : {current_theta[3]:.6f}")
        

if __name__ == '__main__':
    # get data
    bmw_prices = pd.read_csv('data/Continental.csv',
                             sep=';',
                             decimal=',')
    price_array = bmw_prices['Schlusskurs'].astype(float).values
    price_array = np.flip(price_array)
    return_array = compute_log_return(price_array)
    
    # define consts
    risk_premium = 0
    risk_free_rate = 0
    
    # sample calculate with given theta
    # GARCH(1, 1) parameter: (alpha_0, alpha_1, beta_1, initial standard deviation sigma)   
    theta_given = (0.01, 0.2, 0.8, np.std(return_array))

    # sigma_sq_array, xi_array = compute_sigma_sq_historic(theta_given, risk_premium, risk_free_rate, return_array)
    # neg_llh = neg_log_likelihood_GARCH_11(theta_given, risk_premium, risk_free_rate, return_array)
    
    
    # optimization
    theta_hat = estimate_GARCH_11_theta(return_array, risk_free_rate)
    print(theta_hat)