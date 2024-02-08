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


class VaRMonteCarlo:
    def __init__(self, S, mu, sigma, c, n, iterations):
        self.S = S
        self.mu = mu
        self.sigma = sigma
        self.c = c
        self.n = n
        self.iterations = iterations

    def simulation(self):
        rand = np.random.normal(0, 1, [1, self.iterations])

        # equation for the S(t) stock price
        #
        stock_price = self.S * np.exp(self.n * (self.mu - 0.5*self.sigma ** 2) +
                                      self.sigma * np.sqrt(self.n) * rand)

        # we have to sort the stock prices to determine the percentile
        stock_price = np.sort(stock_price)

        # it depends on the confidence level: 95% -> 5 and 99% -> 1
        percentile = np.percentile(stock_price, (1 - self.c) * 100)

        return self.S - percentile


if __name__ == '__main__':
    notional = 1e6 # this is our investment
    confidence_level = 0.95 # confidence level
    days = 10 # 1 day
    iterations = 10000
    ticker = 'HOOD'

    start_date = datetime.datetime(2023,1,1)
    end_date = datetime.datetime(2023, 12, 29)

    stock = download_data(ticker, start_date, end_date)
    stock['return'] = stock[ticker].pct_change()

    # mean, std.
    average_return = np.mean(stock['return'])
    vol = np.std(stock['return']) * np.sqrt(days)

    MC_VaR = []
    for i in range(100):
        model = VaRMonteCarlo(notional, average_return, vol, confidence_level, days, iterations)
        print("Value at risk is: $%.2f" % model.simulation())
        MC_VaR .append(model.simulation())

    plt.plot(MC_VaR)
    plt.show()