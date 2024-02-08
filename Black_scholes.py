from scipy import stats
from numpy import log, exp, sqrt

def call_option_prices(S, E, T, rf, sigma):
    # first calculate d1 and d2 parameters
    d1 = (log(S/E) + (rf + 0.5 * (sigma ** 2)) * T)/(sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    print("The d1 and d2 parameters: %s, %s" % (d1, d2))

    # use the N(X) to calculate the price of the action
    return S*stats.norm.cdf(d1) - E * exp(-rf*T) * stats.norm.cdf(d2)

def put_option_prices(S, E, T, rf, sigma):
    # first calculate d1 and d2 parameters
    d1 = (log(S/E) + (rf + 0.5*(sigma**2))*T)/(sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    print("The d1 and d2 parameters: %s, %s" % (d1, d2))

    # use the N(X) to calculate the price of the action
    return -S * stats.norm.cdf(-d1) + E * exp(-rf * T) * stats.norm.cdf(-d2)




if __name__ == '__main__':
    # underlying stock price at t=0
    S0 = 100
    # strike price
    E = 100
    # risk-free rate
    rf = 0.05
    # maturity
    T = 1
    # volatility
    sigma = 0.2

    print("call option prices according to blackscholes model %s" % call_option_prices(S0, E, T, rf, sigma))
    print("call option prices according to blackscholes model %s" % put_option_prices(S0, E, T, rf, sigma))