import tushare as ts 
import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostRegressor

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='SimSun')  #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  #用来正常显示负号

from sklearn.decomposition import PCA

pd.set_option('display.max_columns', None)

count_row = -1
count_col = -1

value_count_row = 400
test_num = 100
number_of_day_before = 1

point = 0.02

base_point = 1

stock_list = ['000001','000049', '000338', '000002', '000520', '000537', '000540', '000568', '000629', '000636', '000651', '000661']

def timing(stock_code, split_point):
	x_train = []
	temp_x_train = []
	y_train = []
	x_test = []
	temp_x_test = []
	y_test = []
	df = ts.get_hist_data(stock_code)
	#print (df)
	if df is None:
		count_row = -2
	else:
		count_row = -1
	if count_row > -2:
		(count_row, count_col) = df.shape
	if count_row > value_count_row:
		#print (df)
		#print (count_row)
		base_point = df.iloc[count_row - 1, 2]
		df['return'] = np.nan
		for i in range(number_of_day_before, count_row):
			df.iloc[i, 13] = (df.iloc[i - number_of_day_before, 2] - df.iloc[i, 2]) / df.iloc[i, 2]
		#print (df)

		'''
		data = df[test_num : count_row]
		print (data)
		'''

		###################################################
		for i in range(count_row - 1, test_num - 1, -1):
			for j in range(0, count_col):
				temp_x_train.append(df.iloc[i, j])
			x_train.append(temp_x_train)
			temp_x_train = []
			y_train.append(df.iloc[i, count_col])
		'''	
		for i in range(0, len(y_train)):
			if y_train[i] < -1 * point:
				y_train[i] = -1
			elif y_train[i] > point:
				y_train[i] = 1
			else:
				y_train[i] = 0
		'''
		print (x_train)
		MinMax = MinMaxScaler()
		x_train = MinMax.fit_transform(x_train)

		###################################################
		for i in range(test_num - 1, number_of_day_before - 1, -1):
			for j in range(0, count_col):
				temp_x_test.append(df.iloc[i, j])
			x_test.append(temp_x_test)
			temp_x_test = []
			y_test.append(df.iloc[i, count_col])
		'''
		for i in range(0, len(y_test)):
			if y_test[i] < -1 * point:
				y_test[i] = -1
			elif y_test[i] > point:
				y_test[i] = 1
			else:
				y_test[i] = 0
		'''
		x_test = MinMax.transform(x_test)
		###################################################




		estimator = PCA(n_components=5)
		x_train = estimator.fit_transform(x_train)
		x_test = estimator.transform(x_test)

		###################################################

		#model = SVC(C = 1.0, kernel = 'rbf', class_weight = {-1: 4, 0: 1, 1: 4})
		#model = SVR(kernel='rbf', C=1000)
		#model = RandomForestRegressor(n_estimators=50)
		model = AdaBoostRegressor()
		model.fit(x_train, y_train)
		y_predict = model.predict(x_test)
		print ('*****************************')
		print (stock_code)
		print ('y_test')
		print (y_test)
		print ('y_predict')
		print (y_predict)

		time_length = len(y_test)
		sum_test = np.zeros(time_length)
		sum_predict = np.zeros(time_length)
		sum_test[0] = base_point * (1 + y_test[0])
		sum_predict[0] = base_point * (1 + y_predict[0])
		for i in range(1, time_length):
			sum_test[i] = sum_test[i - 1] * (1 + y_test[i])
			sum_predict[i] = sum_predict[i - 1] * (1 + y_predict[i])


		fig,ax = plt.subplots()
		index = range(0, time_length)
		plt.plot(index, sum_test, "x-", label = "test")
		plt.plot(index, sum_predict, "+-", label = "predict")

		# 画买入/卖出点和直线
		pre_min = min(sum_predict)
		pre_max = max(sum_predict)

		minindex = np.where(sum_predict == pre_min)[0][0]
		maxindex = np.where(sum_predict == pre_max)[0][0]

		plt.axvline(x=minindex, c = 'r')
		plt.axvline(x=maxindex, c='g')

		#换买入点到卖出点的直线
		if minindex < maxindex:
			plt.plot([minindex, maxindex], [sum_test[minindex], sum_test[maxindex]], color='b', marker='o')
			profit_prt = cal_percent(sum_test[minindex], sum_test[maxindex])
			plt.title('股票:{3} 第{0}日买入，第{1}日卖出，回测收益率{2}'.format(minindex, maxindex, profit_prt, stock_code))
		else:
			plt.title('预测下跌，不推荐持仓')

		#计算收益率


		plt.legend(bbox_to_anchor=(0.23, 0.97), loc=1, borderaxespad=0.)

		plt.show()


def cal_percent(buy_price, sell_price):
	percent = np.round(sell_price/buy_price - 1, 2)
	return '{0:.0f}%'.format(percent*100)

def timing_package(stock_list):
	for i_stock in stock_list:
		#code = '%06d' % i_stock 
		#print (code)
		timing(i_stock, 0)



timing_package(stock_list)
