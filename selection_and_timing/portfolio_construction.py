import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import pickle

#调用的话  请运行： draw_efficient_frontier(stock_list, iterations, load_data)

#设定各种参数
risk_free_rate=0.03102
iterations = 10000
load_data = False

start = '2009-01-01'
end = '2019-03-01'
#list of stocks in portfolio
stock_list = ('600000.SS',
'600004.SS',
'600009.SS',
'600010.SS',
'600011.SS',
'600015.SS',
'600016.SS',
'600018.SS',
'600019.SS',
'600027.SS')


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
    # plt.scatter(results_frame.stdev,results_frame.ret, \
    #             c=results_frame.sharpe,cmap='RdYlBu')
    # plt.colorbar()
    results_frame.plot.scatter(x='Volatility', y='Returns', c='Sharpe Ratio',
                               cmap='RdYlGn', edgecolors='black', figsize=(15, 8), grid=True)

    # 画最优点和最小方差点
    min_volatility = results_frame['Volatility'].min()
    max_sharpe = results_frame['Sharpe Ratio'].max()

    # use the min, max values to locate and create the two special portfolios
    sharpe_portfolio = results_frame.loc[results_frame['Sharpe Ratio'] == max_sharpe]
    min_variance_port = results_frame.loc[results_frame['Volatility'] == min_volatility]

    plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
    plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200)

    plt.title('Efficient Frontier')
    plt.ylabel('Expected Returns')
    plt.xlabel('Volatility (Std. Deviation)')
    plt.show()
    plt.savefig('有效前沿.jpg')
    plt.close()


def draw_efficient_frontier(stock_list, iterations, load_data=False):
    if load_data:
        print('loading data')
        data = load_stock_data(stock_list, start, end)
    else:
        data = pickle.load(open("yahoo.pkl", "rb"))

    plot_function(data, iterations)

draw_efficient_frontier(stock_list, iterations, load_data)