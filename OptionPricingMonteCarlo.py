import numpy as np

class OptionPricing:

    def __init__(self, S0, E, T, rf, sigma, iterations):
        self.S0 = S0
        self.E = E
        self.T = T
        self.rf = rf
        self.sigma = sigma
        self.iteration = iterations
    def call_option_simulation(self):
        # we have 2 columns: first with 0s the second column will store the payoff
        # we need the first column of 0s: payoff unction is max(0, S-E) for call option
        option_data = np.zeros([self.iteration, 2])

        # dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, (1, self.iteration))

        # calculate the stock price
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma **2) + self.sigma * np.sqrt(self.T) * rand)

        # we need S-E because we have to calculate the max (S-E, 0)

        option_data[:, 1] = stock_price - self.E

        # average for the Monte-Carlo simulation
        # max() returns the max (0, S-E) according to the formula
        average = np.sum(np.amax(option_data, axis=1))/float(self.iteration)

        # have to use the exp(-rT) discount factor
        return np.exp(-1 * self.rf * self.T) * average

    def put_option_simulation(self):
        # we have 2 columns: first with 0s the second column will store the payoff
        # we need the first column of 0s: payoff unction is max(0, S-E) for call option
        option_data = np.zeros([self.iteration, 2])

        # dimensions: 1 dimensional array with as many items as the iterations
        rand = np.random.normal(0, 1, (1, self.iteration))

        # calculate the stock price
        stock_price = self.S0 * np.exp(self.T * (self.rf - 0.5 * self.sigma **2) + self.sigma * np.sqrt(self.T) * rand)

        # we need S-E because we have to calculate the max (S-E, 0)

        option_data[:, 1] = self.E - stock_price

        # average for the Monte-Carlo simulation
        # max() returns the max (0, E-S) according to the formula
        average = np.sum(np.amax(option_data, axis=1))/float(self.iteration)

        # have to use the exp(-rT) discount factor
        return np.exp(-1 * self.rf * self.T) * average

if __name__ == '__main__':
    model = OptionPricing(100, 100, 1, 0.05, 0.2, 10000)

    call_option_value = model.call_option_simulation()
    put_option_value = model.put_option_simulation()

    print("Value of the call option is %.2f" % call_option_value)
    print("Value of the put option is %.2f" % put_option_value)