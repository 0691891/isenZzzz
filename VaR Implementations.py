import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
import datetime
import matplotlib.pyplot as plt

def download_data(stock, start_date, end_date):
    data = {}
    tickers = yf.download(stock, start_date, end_date)
    data[stock] = tickers['Adj Close']
    return pd.DataFrame(data)

def calculate_var(position, c, mu, sigma):

    # inverse cumulative distribution function (cdf -> ppf)
    v = norm.ppf(1-c)

    # calculate VaR
    var = position * (mu - sigma*v)

    return var


if __name__ == '__main__':

    start = datetime.datetime(2023,1,1)
    end = datetime.datetime(2023, 12, 29)

    stock_data = download_data('NVDA', start, end)
    stock_data['returns'] = np.log(stock_data['NVDA']/stock_data['NVDA'].shift(1))
    print(stock_data[1:])

    plt.plot(stock_data['returns'])
    plt.show()

    # this is the investment
    S = 1e6

    # confidence level - this time it is 95%
    c = 0.99

    # we assume that daily returns are normally distributed
    mu = np.mean(stock_data['returns'])
    sigma = np.std(stock_data['returns'])

    print("Value at risk is: %.2f" % calculate_var(S, c, mu, sigma))
