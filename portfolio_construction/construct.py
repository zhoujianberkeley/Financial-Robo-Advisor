import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pickle

#调用的话  请运行： draw_efficient_frontier(stock_list, iterations, load_data)

risk_free_rate=0.03102

# download daily price data for each of the stocks in the portfolio
def load_stock_data(stock_list, start, end):
    data = web.DataReader(stock_list, data_source='yahoo', start=start, end=end)['Adj Close']
    assert data.shape[0] > 0
    data.to_pickle('yahoo.pkl')
    print('save stock historical data')
    return data


def plot_function(data, iterations):
    data.sort_index(inplace=True)

    # convert daily stock prices into daily returns
    returns = data.pct_change()

    # calculate mean daily return and covariance of daily returns
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()

    # set number of runs of random portfolio weights
    num_portfolios = iterations

    # set up array to hold results
    results = np.zeros((3, num_portfolios))

    # efficient frontier

    for i in range(num_portfolios):
        # print(i)
        # select random weights for portfolio holdings
        weights = np.random.random(data.shape[1])
        # rebalance weights to sum to 1
        weights /= np.sum(weights)

        # calculate portfolio return and volatility
        portfolio_return = np.sum(mean_daily_returns * weights) * 252  # need futhur considertation
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        # store results in results array
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        # store Sharpe Ratio (return / volatility) - risk free rate element excluded for simplicity
        results[2, i] = (results[0, i] - risk_free_rate) / results[1, i]

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(results.T, columns=['Returns', 'Volatility', 'Sharpe Ratio'])

    # create scatter plot coloured by Sharpe Ratio
    plt.style.use('seaborn-dark')

    results_frame.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                               cmap='RdYlGn', edgecolors='black', figsize=(15, 8), grid=True)

    # 画最优点和最小方差点
    min_volatility = results_frame['Volatility'].min()
    max_sharpe = results_frame['Sharpe Ratio'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = results_frame.loc[results_frame['Sharpe Ratio'] == max_sharpe]
    minvar_portfolio = results_frame.loc[results_frame['Volatility'] == min_volatility]

    plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
    plt.scatter(x=minvar_portfolio['Volatility'], y=minvar_portfolio['Returns'], c='blue', marker='D', s=200)

    plt.title('Efficient Frontier')
    plt.ylabel('Expected Returns')
    plt.xlabel('Volatility (Std. Deviation)')
    return results_frame


def rank_point(data, rank):
    min_volatility = data['Volatility'].min()
    max_sharpe = data['Sharpe Ratio'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = data.loc[data['Sharpe Ratio'] == max_sharpe]
    minvar_portfolio = data.loc[data['Volatility'] == min_volatility]

    sharpe_vol = sharpe_portfolio['Volatility'].values[0]
    min_vol = minvar_portfolio['Volatility'].values[0]
    vol_dif = (sharpe_vol - min_vol) / 5

    vol = vol_dif * rank + min_volatility

    rank_point = data[data['Volatility'] < vol].sort_values(by='Sharpe Ratio', ascending=False)
    return rank_point.iloc[0, :]

def plot_Point(point):
    plt.scatter(x=point['Volatility'], y=point['Returns'], c='orange', marker='D', s=200)


def load_data_func(stock_list, load_data, start, end):
    if load_data:
        print('loading data')
        data = load_stock_data(stock_list, start, end)
    else:
        data = pickle.load(open("../yahoo.pkl", "rb"))
    return data


def draw_norank(stock_list, iterations, load_data=False, start = None, end = None):
    '''
    画没有rank的ef
    '''
    data = load_data_func(stock_list, load_data, start, end)
    plot_function(data, iterations)

    plt.show()
    plt.close()


def draw_rank(stock_list, iterations, rank, load_data=False, start = None, end = None):
    '''
    画有用户等级rank的有效前沿
    '''
    if load_data:
        print('loading data')
        data = load_stock_data(stock_list, start, end)
    else:
        data = pickle.load(open("../yahoo.pkl", "rb"))

    result_frame = plot_function(data, iterations)
    point = rank_point(result_frame, rank)
    plot_Point(point)

    plt.show()
    plt.close()
