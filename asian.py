

#=====================================================================
# import datetime as dt
# import yfinance as yf
# import pandas as pd
# import numpy as np
# from scipy.stats import norm
# =============================================================================


class Asian (object):
    
    def __init__(self,T,S,payoff):
        self.time = T #no. of business days before expiry
        self.stock = S #ticker in yahoo finance
        self.avg = None
        self.payoff = payoff #payoff is a function of avg and stock at expiry time
        
    def get_data(self,start,end,interval):
        "get historical data for the stock; \
         start-end format: 'yyyy-mm-dd' \
         interval acceptable inputs: 1m,2m,5m,15m,30m,1h,1d..."
        return yf.download(self.stock, start, end, interval=interval)
    
    def estimated_drift(self,interval, parameter):
        start = dt.today() - dt.timedelta(days = 3*self.time)
        end = dt.today()
        df = self.get_data(start,end,interval)
        df['rolling_avg'] = df['Close'].rolling(parameter).mean()
        df.dropna()
        period_length = 3*self.time - parameter*conv_interval(interval)
        val_1 = df['rolling_avg'][1]
        val_2 = df['rolling_avg'][-1]
        return np.log(val_2 - val_1)/period_length
        
        
    def estimated_volatility_brownian(self, interval, drift):
        "returns estimated volatility under Brownian assumption based on historical \
         data over a period 3T. \
         "
        start = (dt.date.today() - dt.timedelta(days = 3*self.time)).strftime("%Y-%m-%d")
        end = (dt.date.today()).strftime("%Y-%m-%d")
        df = get_data(self,start,end,interval)
        df['diffs'] = ((df['Close'] - df['Close'].shift(1))**2)
        std = np.sqrt((df['diffs'].sum()/len(diffs.dropna())))
        return len(diffs)*std/3
    
    def estimated_volatility_european(self, european_price, strike, time, current_stock_price, dividend, risk_free_rate, precision):
        "Recovers implied volatility (from a European call) under BS assumption using Newton Raphson's method"
        "Note that Vega is positive so we can use NR method effectively"
        C, S, T, K, y, r = european_price, strike, time, current_stock_price, dividend, risk_free_rate
        
        def N(sign, sig):
            if sign == 'plus':
                d = np.log(S/K) + (r-y+(sig**2)/2)*T
                return norm.cdf(d)
            elif sign == 'minus':
                d = np.log(S/K) + (r-y-(sig**2)/2)*T
                return norm.cdf(d)
            
        def vega(sig):
            d = np.log(S/K) + (r-y+(sig**2)/2)*T
            return np.sqrt(T/(2*np.pi))*S*np.exp(-d**2/2)
            
        sigma = (0, np.sqrt(2*np.pi/T)*C/S) 
        while abs(sigma[1] - sigma[0]) > precision:
           sigma = (sigma[1], sigma[1] - ((S*np.exp(-y*T)*N('plus',sigma[1]) - K*np.exp(-r*T)*N('minus',sigma[1])) - C)/(vega(sigma[1])))
        return sigma[1]
        
        
    
    def Montecarlo_pricing(self, mu, sigma, intervals, simulations):
        "inputs: \
         mu = continuosly compounded drift rate (ideally estimated with method above) \
         sigma = volatility over T (ideally estimated with method above, multiplied by S*T) \
         intervals = no. of intervals over T used in the discertization of the brownian motion of S(t) \
         output: \
         estimated value of the option according to the model \
         "
        estimated_payoff = [None]*simulations
        for i in range(simulations) : 
            l = [None]*intervals
            l[0] = yf.download(self.stock,start=(dt.date.today()).strftime("%Y-%m-%d"), end=(dt.date.today()).strftime("%Y-%m-%d"))['Close'][-1]
            for j in range(1,intervals):
                l[j] = l[j-1]*exp(mu/intervals)+sigma/intervals*np.random.normal(1)
            average = sum(l)/intervals
            estimated_payoff[i] = self.payoff(average,l[-1])
        return sum(estimated_payoff)/simulations
            
            
    def conv_interval (time):
        'converts yfinance time input into days'
        if 'm' in time:
            return (int(time[:-1])/1440)
        elif 'h' in time:
            return (int(time[:-1])/24)
        else:
            return int(time[:-1])
                     
        

       
    