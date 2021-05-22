# Pricing-a-Garch-model

Considering a GARCH model without risk premium (λ = 0).

I Made a program to compute the price of a Call option using a GARCH Model. I Used a Monte Carlo method to evaluate the expectation with respect to the law of the GARCH process. For the calibration of the GARCH model, I used typical values of the parameters for financial data on a daily scale (the model is not estimated on real data). 
For the time to expiration date, I choosed T = 250, corresponding to one year.

Numerically I studied how the price the option depends on the value of the ratio S/K, where S is the price of the underlying asset at time 0 and K is the strike price of the option (the ratio S/K could range from 0.5 to 2).

Finally, I computed the values of the implied volatility for the different values of the moneyness S/K.
