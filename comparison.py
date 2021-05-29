import math

import numpy as np
import pandas as pd

import garch as g
import garch_const_lambda as gc
import pricing as p


def compare(theta_hat, 
            risk_free_rate,
            asset_price_t,
            t=0, 
            T=90, 
            moneyness=1.0, 
            conditional_volatility_ratio=1.0):
    # some consts
    a0, a1, b1, risk_premium, sigma_0 = theta_hat
    num_periods = T - t
    num_simulations = 50000
    
    strike_price = 100
    asset_price_t = strike_price * moneyness
    
    # some relevant computation
    BS_sigma_sq = a0 / (1 - a1 - b1) # Black scholes' sigma squared
    GARCH_stationary_variance = a0 / (1 - (1 + risk_premium ** 2) * a1 - b1) # stationary variance under GARCH and Local Risk Neutralization
    GARCH_sigma_t = conditional_volatility_ratio * np.sqrt(GARCH_stationary_variance) # conditional variance
    print("GARCH_sigma_t", GARCH_sigma_t)
    GARCH_asset_price_array_T, _, _ = p.compute_GARCH_price(theta=theta_hat,
                                                            num_periods=num_periods,
                                                            init_price=asset_price_t,
                                                            init_sigma=GARCH_sigma_t,
                                                            risk_free_rate=risk_free_rate,
                                                            num_simulations=num_simulations)
    GARCH_asset_price_T = np.mean(GARCH_asset_price_array_T)
    print(GARCH_asset_price_T)
    # strike_price = GARCH_asset_price_T / moneyness
    
    # GARCH option pricing
    GARCH_call_price, GARCH_call_price_array = p.compute_GARCH_call_price(sim_price_array=GARCH_asset_price_array_T,
                                                                          strike_price=strike_price,
                                                                          num_periods=num_periods,
                                                                          risk_free_rate=risk_free_rate)
    GARCH_call_delta, GARCH_call_delta_array = p.compute_GARCH_delta(sim_price_array=GARCH_asset_price_array_T,
                                                                     strike_price=strike_price,
                                                                     num_periods=num_periods,
                                                                     current_price=asset_price_t,
                                                                     risk_free_rate=risk_free_rate)
    BS_call_price = p.compute_BS_call_price(sigma_sq=BS_sigma_sq,
                                            current_price=asset_price_t,
                                            strike_price=strike_price,
                                            risk_free_rate=risk_free_rate,
                                            num_periods=num_periods)
    BS_call_delta = p.compute_BS_delta(sigma_sq=BS_sigma_sq,
                                       current_price=asset_price_t,
                                       strike_price=strike_price,
                                       risk_free_rate=risk_free_rate,
                                       num_periods=num_periods)
    
    # construct comparison result
    # option price
    price_bias_percent_mean = np.mean((GARCH_call_price_array - BS_call_price) / BS_call_price * 100)
    price_bias_percent_std = np.std((GARCH_call_price_array - BS_call_price) / BS_call_price)
    # option delta
    delta_bias_percent_mean = np.mean((GARCH_call_delta_array - BS_call_delta) / BS_call_delta * 100)
    delta_bias_percent_std = np.std((GARCH_call_delta_array - BS_call_delta) / BS_call_delta)
    # result as dict for easy DF later
    result = {
        "t": t,
        "T": T,
        "moneyness": moneyness,
        "conditional_volatility_ratio": conditional_volatility_ratio,
        "GARCH_asset_price": GARCH_asset_price_T,
        # "strike_price": strike_price,
        # "GARCH_stationary_sigma": np.sqrt(GARCH_stationary_variance),
        # "GARCH_sigma_t": GARCH_sigma_t,
        "price_BS": BS_call_price,
        "price_GARCH": GARCH_call_price,
        "price_bias_mean": price_bias_percent_mean,
        "price_bias_std": price_bias_percent_std,
        "delta_BS": BS_call_delta,
        "delta_GARCH": GARCH_call_delta,
        "delta_bias_mean": delta_bias_percent_mean,
        "delta_bias_std": delta_bias_percent_std,
    }
    
    return result


def main():
    # get data, using BMW data
    price_df = pd.read_csv('data/BMW.csv',
                             sep=';',
                             decimal=',',
                             parse_dates=['Datum'])
    
    price_array = price_df['Schlusskurs'].astype(float).values
    price_array = np.flip(price_array)[-500:]
    date_array = np.flip(price_df['Datum'].values)[-500:]
    print(f'Start date: {np.min(date_array)}')
    print(f'End date: {np.max(date_array)}')
    print(f'Duration: {len(date_array)}')
    
    return_array = g.compute_log_return(price_array)
    
    # define some consts
    RISK_PREMIUM_AS_CONSTANT = True
    risk_free_rate = 0.05 / 365
    risk_premium = 0.001
    
    # GARCH(1, 1) parameter estimation
    # GARCH(1, 1) parameter: (alpha_0, alpha_1, beta_1, lambda, sigma)   
    if RISK_PREMIUM_AS_CONSTANT:
        theta_hat = gc.estimate_GARCH_11_theta(return_array, risk_premium, risk_free_rate)
        a0, a1, b1, sigma_0 = theta_hat
        theta_hat = (a0, a1, b1, risk_premium, sigma_0) # rearrange for easier input in later functions
    else:
        theta_hat = g.estimate_GARCH_11_theta(return_array, risk_free_rate)
    
    # set up context for comparison
    t = 0 # current time wrt call option creation date
    T_comp = (30, 90, 180) # option lengths
    moneyness_comp = (.8, .9, .95, 1.0, 1.05, 1.1, 1.2) # asset price at T / strike price
    volatility_ratio = (0.8, 1.0, 1.2) # sqrt(h_t) / stationary sigma
    settings = []
    for _T in T_comp:
        for _m in moneyness_comp:
            for _v in volatility_ratio:
                settings.append((_T, _m, _v))
    
    result_list = []
    for setting in settings:
        T, moneyness, conditional_volatility_ratio = setting
        num_periods = T - t # time-to-maturity
        asset_price_t = price_array[-num_periods]
    
        res = compare(theta_hat=theta_hat,
                    risk_free_rate=risk_free_rate,
                    asset_price_t=asset_price_t,
                    t=t,
                    T=T,
                    moneyness=moneyness,
                    conditional_volatility_ratio=conditional_volatility_ratio)
        print(res)
        result_list.append(res)
    
    result_df = pd.DataFrame(result_list)
    result_df = result_df.groupby(['T', 'moneyness', 'conditional_volatility_ratio', 'price_BS', 'delta_BS']).mean()
    print(result_df)
    result_df = (result_df
                .reset_index(2)
                .drop('t', axis=1)
                .pivot(columns='conditional_volatility_ratio')
                .reorder_levels([1, 0], axis=1)
                .sort_index(axis=1, level=0)
                )
    print(result_df.filter(like='0.8', axis=1))
    print(result_df.filter(like='1.0', axis=1))
    print(result_df.filter(like='1.2', axis=1))
    
    # GARCH_implied_volatility = p.compute_BS_implied_volatility(call_option_price=call_price,
    #                                                         current_price=price_at_t,
    #                                                         strike_price=strike_price,
    #                                                         risk_free_rate=risk_free_rate,
    #                                                         num_periods=T - t)
    # print(GARCH_implied_volatility)
    
    
    # result_df.to_csv('data/result.csv', sep='\t', index=True)
    return result_df

if __name__ == "__main__":
    a = main()
