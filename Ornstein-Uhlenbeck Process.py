import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal

def generate_process(dt = 0.1, theta = 1.2, mu = 0.5, sigma = 0.3, n =10000):
    # x(t=0) = 0 and initialize x(t) with zeros
    x = np.zeros(n)

    for t in range(1, n):
        x[t] = x[t-1] + theta * (mu - x[t-1])*dt + sigma * normal(0, np.sqrt(dt)) # normal(mean, std)
    return x

def plot_process(x):
    plt.plot(x)
    plt.xlabel('time')
    plt.ylabel('x(t)')
    plt.title('Ornstein-Uhlenbeck Process')
    plt.show()

def vasicek_model(r0, kappa, theta, sigma, T=1, N=1000):

    dt = T/float(N)
    t = np.linspace(0, T, N+1)
    rates = [r0]

    for _ in range(N):
        dr = kappa*(theta - rates[-1])*dt + sigma * normal(0, np.sqrt(dt))
        rates.append(rates[-1]+dr)

    return t, rates

def plot_vasicek_model(t, r):
    plt.plot(t, r)
    plt.xlabel('time')
    plt.ylabel('Interest Rate r(t)')
    plt.title('Vasicek Model')
    plt.show()

if __name__ == '__main__':
    data = generate_process()
    plot_process(data)

    times, rates = vasicek_model(1.3, 0.9, 1.4, 0.85)
    plot_vasicek_model(times, rates)
