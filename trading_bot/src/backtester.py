import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class Backtester:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.portfolio = {
            'cash': initial_capital,
            'positions': {},
            'history': []
        }
        self.trades = []
        self.performance_metrics = {}
        
    def run_backtest(self, agent, data, start_date=None, end_date=None):
        """Run backtest simulation"""
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]
            
        # Filter data for backtest period
        mask = (data.index >= start_date) & (data.index <= end_date)
        backtest_data = data[mask].copy()
        
        # Run simulation
        for timestamp in backtest_data.index:
            current_data = backtest_data.loc[timestamp]
            
            # Get agent's action
            state = self._get_state(current_data)
            action = agent.act(state)
            
            # Execute trade
            self._execute_trade(action, current_data)
            
            # Update portfolio history
            self._update_portfolio_history(timestamp, current_data)
            
        # Calculate performance metrics
        self._calculate_performance_metrics(backtest_data)
        
        return self.performance_metrics
    
    def _execute_trade(self, action, data):
        """Execute trading action"""
        current_price = data['Close']
        
        if action == 1:  # Buy
            shares_to_buy = self.portfolio['cash'] // current_price
            cost = shares_to_buy * current_price
            
            if shares_to_buy > 0:
                self.portfolio['cash'] -= cost
                self.portfolio['positions']['shares'] = shares_to_buy
                self.trades.append({
                    'action': 'buy',
                    'price': current_price,
                    'shares': shares_to_buy,
                    'cost': cost
                })
                
        elif action == 2:  # Sell
            if 'shares' in self.portfolio['positions']:
                shares = self.portfolio['positions']['shares']
                revenue = shares * current_price
                
                self.portfolio['cash'] += revenue
                self.portfolio['positions'] = {}
                self.trades.append({
                    'action': 'sell',
                    'price': current_price,
                    'shares': shares,
                    'revenue': revenue
                })
    
    def _update_portfolio_history(self, timestamp, data):
        """Update portfolio history"""
        current_price = data['Close']
        position_value = 0
        
        if 'shares' in self.portfolio['positions']:
            position_value = self.portfolio['positions']['shares'] * current_price
            
        total_value = self.portfolio['cash'] + position_value
        
        self.portfolio['history'].append({
            'timestamp': timestamp,
            'cash': self.portfolio['cash'],
            'position_value': position_value,
            'total_value': total_value
        })
    
    def _calculate_performance_metrics(self, data):
        """Calculate comprehensive performance metrics"""
        portfolio_values = pd.DataFrame(self.portfolio['history'])
        returns = portfolio_values['total_value'].pct_change()
        
        # Basic metrics
        self.performance_metrics['total_return'] = (portfolio_values['total_value'].iloc[-1] / 
                                                  self.initial_capital - 1)
        self.performance_metrics['sharpe_ratio'] = self._calculate_sharpe_ratio(returns)
        self.performance_metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_values['total_value'])
        
        # Trading metrics
        self.performance_metrics['total_trades'] = len(self.trades)
        self.performance_metrics['win_rate'] = self._calculate_win_rate()
        
        # Risk metrics
        self.performance_metrics['volatility'] = returns.std() * np.sqrt(252)
        self.performance_metrics['var_95'] = returns.quantile(0.05)
        self.performance_metrics['var_99'] = returns.quantile(0.01)
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Calculate Sharpe ratio"""
        excess_returns = returns - risk_free_rate/252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown"""
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return drawdown.min()
    
    def _calculate_win_rate(self):
        """Calculate win rate of trades"""
        if not self.trades:
            return 0
            
        profitable_trades = sum(1 for trade in self.trades if 
                              'revenue' in trade and trade['revenue'] > trade.get('cost', 0))
        return profitable_trades / len(self.trades)
    
    def plot_results(self):
        """Plot backtest results"""
        portfolio_values = pd.DataFrame(self.portfolio['history'])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
        
        # Portfolio value over time
        ax1.plot(portfolio_values['timestamp'], portfolio_values['total_value'])
        ax1.set_title('Portfolio Value Over Time')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Value ($)')
        
        # Returns distribution
        returns = portfolio_values['total_value'].pct_change()
        returns.hist(ax=ax2, bins=50)
        ax2.set_title('Returns Distribution')
        ax2.set_xlabel('Return')
        ax2.set_ylabel('Frequency')
        
        # Drawdown
        peak = portfolio_values['total_value'].expanding(min_periods=1).max()
        drawdown = (portfolio_values['total_value'] - peak) / peak
        ax3.fill_between(portfolio_values['timestamp'], drawdown, 0, color='red', alpha=0.3)
        ax3.set_title('Drawdown')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown')
        
        plt.tight_layout()
        plt.show() 