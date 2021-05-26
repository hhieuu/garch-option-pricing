import math

import numpy as np
from numba import jit
from utils import normal_pdf
from scipy.stats import norm


@jit(nopython=True)
def compute_GARCH_price(theta, 
                        num_periods, 
                        init_price, 
                        init_sigma, 
                        risk_free_rate, 
                        num_simulations=50000):
    """
    Compute asset price at period t + s (s periods ahead) given estimated theta values and initial price at t
    using Monte Carlo simulation
    
    Param
    -----
    theta: Tuple(Float, Float, Float, Float, Float)
        GARCH(1, 1) parameter: (alpha_0, alpha_1, beta_1, risk premium lambda, standard deviation sigma)
        
    num_periods: Int
        Number of steps ahead for which price is calculated.
        If we are at time t, and would like to compute price at T, then num_periods = T - t
        
    init_price: Float
        Price at time t
        
    init_sigma: Float
        Conditional standard deviation at time t
        
    risk_free_rate: Float
        Return of risk-free asset. If theta is estimated with a certain risk-free-rate,
        then that rate should be given here.
        
    num_simulation: Int
        Number of Monte Carlo simulations to be made
        
    Return
    ------
    Tuple(ndarray, ndarray, ndarray):
        A tuple of size 3, where the elements are, respectively 
        (simulated prices, simulated conditional variance, simulated innovation).
        Each ndarray has shape (n, ), where n is the number of MC simulations made.
        
    """
    a0, a1, b1, lamb, sigma0 = theta
    
    # set initial values
    # variance sigma_sq_t
    sigma_sq_array = np.zeros(shape=(num_periods, num_simulations))
    sigma_sq_array[0, :] = init_sigma ** 2
    # innovation xi_t
    xi_array = np.random.standard_normal(size=(num_periods, num_simulations))
    xi_array[0, :] = xi_array[0, :] * init_sigma
    
    # compute sigma_sq and xi iteratively for num_period
    for t in range(1, num_periods):
        sigma_sq_array[t, :] = a0 \
                             + a1 * (xi_array[t - 1, :] - lamb * np.sqrt(sigma_sq_array[t - 1, :])) ** 2 \
                             + b1 * sigma_sq_array[t - 1, :] # sigma_sq at t
        xi_array[t, :] = xi_array[t, :] * np.sqrt(sigma_sq_array[t, :]) # xi at t
    
    # compute s-period-ahead price for num_simulations
    price_array = init_price * np.exp(num_periods * risk_free_rate \
                                      - 1/2 * np.sum(sigma_sq_array, axis=0) \
                                      + np.sum(xi_array, axis=0))
    
    return price_array, sigma_sq_array, xi_array


@jit(nopython=True)
def compute_GARCH_call_price(sim_price_array,
                             strike_price,
                             num_periods,
                             risk_free_rate):
    """
    Compute price of Call option
    """
    expected_price_array = np.maximum(sim_price_array - strike_price, 0)
    call_price_array = np.exp(-num_periods * risk_free_rate) * expected_price_array
    call_price = np.mean(call_price_array)
    return call_price, call_price_array


@jit(nopython=True)
def compute_GARCH_delta(sim_price_array,
                        strike_price,
                        current_price,
                        num_periods,
                        risk_free_rate):
    """
    Compute Greek Delta
    """
    mask_array = np.where(sim_price_array >= strike_price, 1, 0)
    expected_value_array = sim_price_array / current_price * mask_array
    delta_array = np.exp(-num_periods * risk_free_rate) * expected_value_array
    delta = np.mean(delta_array)
    return delta, delta_array


# Black Scholes formula
# @jit(nopython=True)
def compute_d_t(theta,
                current_price,
                strike_price,
                risk_free_rate,
                num_periods):
    """
    Compute d_t for Black-Scholes option pricing formular in GARCH framework
    """
    a0, a1, b1, lamb, sigma0 = theta
    
    sigma_sq = a0 / (1 - a1 - b1)
    d_t_denom = np.log(current_price / strike_price) + (risk_free_rate + sigma_sq / 2) * num_periods
    d_t = d_t_denom / np.sqrt(sigma_sq * num_periods)
    
    return d_t, sigma_sq
    

# @jit(nopython=True)
def compute_BS_call_price(theta,
                          current_price,
                          strike_price,
                          risk_free_rate,
                          num_periods):
    """
    Compute call price using Black-Scholes option pricing model in GARCH framework
    """
    d_t, sigma_sq = compute_d_t(theta, current_price, strike_price, risk_free_rate, num_periods)
    call_price = current_price * norm.cdf(d_t) \
                - np.exp(-num_periods * risk_free_rate) \
                * strike_price \
                * norm.cdf(d_t - np.sqrt(sigma_sq * num_periods))
    
    return call_price
    
    
# @jit(nopython=True)
def compute_BS_delta(theta,
                     current_price,
                     strike_price,
                     risk_free_rate,
                     num_periods):
    """
    Compute call price using Black-Scholes option pricing model in GARCH framework
    """
    d_t, _ = compute_d_t(theta, current_price, strike_price, risk_free_rate, num_periods)
    return norm.cdf(d_t)


if __name__ == "__main__":
    # theta_hat = (0.01, 0.15, 0.7, 0.02, 0.02)
    theta_name = ("alpha_0", "alpha_1", "beta_1", "lambda", "sigma_0")
    theta_hat = (1.00000000e-04, 2.36543920e-01, 5.80094456e-01, 8.17288812e-01, 1.14291781e-02)
    price_at_t = 100
    strike_price = 105
    sigma_at_t = 0.04
    risk_free_rate = 0.001
    num_simulations = 100000
    t = 0
    T = 90
    
    print()
    print(f'=== EUROPEAN CALL OPTION ================')
    print(f"Current assest price      S_t : {price_at_t}")
    print(f"Strike price              K   : {strike_price}")
    print(f"Current time              t   : {t}")
    print(f"Time to Maturity          T   : {T}")
    print(f"Risk free rate            r   : {risk_free_rate}")
    print(f"Number of MC simulations  n   : {num_simulations}")
    print()
    print(f"=== ESTIMATED PARAMETERS ================")
    w_len = len(theta_name[-1])
    for i in range(len(theta_name)):
        print(f"{theta_name[i]}{' ' * (w_len - len(theta_name[i]))} = {theta_hat[i]:.4f}")
        
    print()
    print(f"=== GARCH PRICING RESULTS ===============")
    sim_price_array, sim_sigma_sq_array, sim_xi_array = compute_GARCH_price(theta=theta_hat,
                                                                      num_periods=T - t,
                                                                      init_price=price_at_t,
                                                                      init_sigma=sigma_at_t,
                                                                      risk_free_rate=risk_free_rate,
                                                                      num_simulations=num_simulations)

    print(f"GARCH asset price at T = {T}   : {np.mean(sim_price_array):.4f}")
    
    call_price = compute_GARCH_call_price(sim_price_array,
                                          strike_price,
                                          T - t,
                                          risk_free_rate)
    print(f"GARCH call price              : {call_price:.4f}")
    
    call_delta = compute_GARCH_delta(sim_price_array,
                                     strike_price,
                                     price_at_t,
                                     T - t,
                                     risk_free_rate)
    print(f"GARCH call delta              : {call_delta:.4f}")
    
    print()
    print(f"=== BS PRICING RESULTS ==================")
    bs_call_price = compute_BS_call_price(theta_hat,
                                          price_at_t,
                                          strike_price,
                                          risk_free_rate,
                                          T - t)
    print(f"Black-Scholes call price      : {bs_call_price:.4f}")
    
    bs_delta = compute_BS_delta(theta_hat,
                                price_at_t,
                                strike_price,
                                risk_free_rate,
                                T - t)
    print(f"Black-Scholes delta           : {bs_delta:.4f}")
    
    
    
    
    
    
