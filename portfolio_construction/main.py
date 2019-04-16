import portfolio_construction.construct as pc

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
rank = 4


def first_run(stock_list, iterations, load_data = False, start = None, end = None):
    pc.draw_norank(stock_list, iterations, load_data)


def second_run(stock_list, iterations, rank):
    pc.draw_rank(stock_list, iterations, rank, load_data)

test = 0
if test:
    # first_run(stock_list, iterations, False)
    #
    # first_run(stock_list, iterations, True, start, end)

    second_run(stock_list, iterations, rank)

