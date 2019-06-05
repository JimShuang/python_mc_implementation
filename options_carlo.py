# implementation of monte carlo simulations on underlying stock of a European call option
# using monte-carlo risk neutral valuation

# The risk neutral approach is when the price of an option on a stock
# grows at the same risk-free rate as the risk-free rate which is used to discount it.

#Use black scholes to calculate price and use as benchmark
#simulate S (price of underlying asset)
#Calculate Payoff = Max(S(t) - K, 0)
# Sum pay-offs and discount them to risk-free rate
# Calculate price of the option from the discounted sum of payoffs.

#price calculated at step 5 should come very close to the price of the option calculated in step 1
# weiner process   ε√Δt where ε is a standard normal variable
# Ito lemma   δS(t) = Drift + Uncertainty
# Geometric Brownian Motion (GBM)

# σ is the volatility
#Aim is calculating the option price by simulating the Wiener process.

# solution of the stochastic differential equation of GBM by using Ito’s lemma:
# S(t) = S(0) exp[(r-0.5σ²)t + σ√tε)]

# simplified design so it is extensible and more trades can be eaily added, as well as adding advanced models

import numpy as np


class Config:
    # stores configuration parameters
    def __init__(self, num_scenarios, num_steps):
        self.num_scenarios = num_scenarios
        self.num_steps = num_steps


class TradeOption:
    # stores option details
    def __init__(self, stock_price, strike_price, risk_free_rate, volatility, maturity_time):
        self.stock_price = stock_price
        self.strike_price = strike_price
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.maturity_time = maturity_time


class GBM:
    # Geometric Brownian Motion Simulator
    # Class will contain the GBM diffeq, it will take in trades and generate simulated stock prices
    def __init__(self, config):
        self.config = config

    #simulate risk factors using GBM stochastic diffeq
    def simulation_of_risk_factors(self, trade):
        prices = []
        # European option only concerns 1 time_step
        time_step = 1
        for simulation_num in range(self.Config.num_scenarios):
            random_norm_num = np.random.normal(0, 1)
            drift = trade.risk_free_rate - 0.5*(trade.volatility**2)*time_step
            uncertainty = trade.volatility*np.sqrt(time_step)*random_norm_num
            price = trade.stock_price*np.exp(drift+uncertainty)
            prices.append(price)
        return prices


class Option_Payoff_Pricing:
    # option payoff pricing
    # generates average call option payoff taking in the trade as a parameter

    def get_price(self, trade, prices_per_simulation):
        total_payoffs = 0
        total_simulations = len(prices_per_simulation)
        for i in range(total_simulations):
            price = prices_per_simulation[i]
            payoff = price - trade.strike_price
            if (payoff>0):
                total_payoffs = total_payoffs+payoff
        price_discounted = (np.exp(-1.0*trade.risk_free_rate * trade.maturity_time)*payoffs)
        average_payoff = price_discounted/total_simulations
        return average_payoff


class MC_Simulator:
    # calls simulation model to simulate the risk factor (in this example it's the stock price)
    # passes simulated risk factors to the payoff pricer to value the trade

    # instantiate with configuration and model
    def __init__(self, config, model):
        self.config = config
        self.model = model

    # similate the trade and give the price
    def Simulation(self, trade, trade_pricer):
        prices_per_simulation = self.model.simulation_of_risk_factors(trade)
        price = trade_pricer.get_price(trade, prices_per_simulation)
        return price


def plot_scenarios(prices_per_simulation):
    x = []
    y = []
    for i in prices_per_simulation:
        y.append(i)
        y.append(trade.stock_price)
        x.append(1)
        x.append(0)
        plt.plot(x, y)
    plt.ylabel("Stock Value")
    plt.xlabel("Step")
    plt.show()



# pricing option with black scholes with characteristics:
    # S = 200, K = 200, T = 1 year, Volatility = 10%, Risk-Free Rate = 15%
def Main():
    config = Config(10000, 1) #1000 scenarios & steps
    trade = TradeOption(200, 200, 0.15, 0.1, 1)
    model = GBM(config)
    tradePricer = Option_Payoff_Pricing()
    simulator = MC_Simulator(config, model)
    price = simulator.Simulation(trade, tradePricer)
    print(price)
