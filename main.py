import torch
import numpy as np
import backtrader as bt
from agent import *
from datetime import datetime
import yfinance as yf

class DQNStrategy(bt.Strategy):
    params = dict(
        state_dim=10,
        action_dim=3
    )
    
    def __init__(self):
        self.agent = DQNAgent(self.params.state_dim, self.params.action_dim)
        self.dataclose = self.datas[0].close
        self.order = None
    
    def next(self):
        state = self.get_state()
        action = self.agent.act(state)
        reward = 0
        
        if action == 1 and self.order is None:
            self.order = self.buy()
        elif action == 2 and self.order is None:
            self.order = self.sell()
        elif action == 0 and self.order:
            self.close()
            reward = self.dataclose[0] - self.order.executed.price if self.order.isbuy() else self.order.executed.price - self.dataclose[0]
            self.order = None

        next_state = self.get_state()
        self.agent.remember(state, action, reward, next_state)
        self.agent.replay()
        self.agent.update_target_network()

    def get_state(self):
        # Current features
        close = self.dataclose[0]
        high = self.datas[0].high[0]
        low = self.datas[0].low[0]
        open_price = self.datas[0].open[0]
        
        # Additional features: Simple Moving Averages for different periods
        sma5 = sum(self.dataclose.get(size=5)) / 5
        sma10 = sum(self.dataclose.get(size=10)) / 10
        sma15 = sum(self.dataclose.get(size=15)) / 15
        sma20 = sum(self.dataclose.get(size=20)) / 20
        sma25 = sum(self.dataclose.get(size=25)) / 25
        sma30 = sum(self.dataclose.get(size=30)) / 30

        # Combine all features into a single array
        state = np.array([close, high, low, open_price, sma5, sma10, sma15, sma20, sma25, sma30])
        
        return torch.tensor(state, dtype=torch.float).unsqueeze(0)

# Main function to run Backtrader
if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(DQNStrategy)
    
    data = bt.feeds.PandasData(dataname=yf.download('AAPL', '2015-07-06', '2021-07-01', auto_adjust=True))
    cerebro.adddata(data)
    
    initial_cash = 100000  # Set your initial cash
    cerebro.broker.set_cash(initial_cash)
    
    cerebro.run()
    cerebro.plot()
    
    final_value = cerebro.broker.getvalue()
    profit = final_value - initial_cash
    print(f"Initial Cash: ${initial_cash:.2f}")
    print(f"Final Portfolio Value: ${final_value:.2f}")
    print(f"Profit: ${profit:.2f}")
    cerebro.addstrategy(DQNStrategy)
    cerebro.addstrategy(DQNStrategy)
    
    data = bt.feeds.PandasData(dataname=yf.download('AAPL', '2015-07-06', '2021-07-01', auto_adjust=True))
    cerebro.adddata(data)
    
    cerebro.broker.set_cash(100000)
    cerebro.run()
    cerebro.plot()
