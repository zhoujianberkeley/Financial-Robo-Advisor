import tushare as ts 
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)

count_row = -1
count_col = -1

value_count_row = 400
test_num = 150
number_of_day_before = 30

point = 0.08

def timing(stock_code, split_point):
	x = []
	temp_x = []
	y = []
	df = ts.get_hist_data(stock_code)
	if df is None:
		count_row = -2
	else:
		count_row = -1
	if count_row > -2:
		(count_row, count_col) = df.shape
	if count_row > value_count_row:
		#print (df)
		#print (count_row)
		df['return'] = np.nan
		for i in range(number_of_day_before, count_row):
			df.iloc[i, 13] = (df.iloc[i - number_of_day_before, 2] - df.iloc[i, 2]) / df.iloc[i, 2]
		#print (df)

		'''
		data = df[test_num : count_row]
		print (data)
		'''
		for i in range(count_row - 1, test_num - 1, -1):
			for j in range(0, count_col):
				temp_x.append(df.iloc[i, j])
			x.append(temp_x)
			temp_x = []
			y.append(df.iloc[i, count_col])

		for i in range(0, len(y)):
			if y[i] < -1 * point:
				y[i] = 0
			elif y[i] > point:
				y[i] = 2
			else:
				y[i] = 1
		'''
		print (x)
		print (y)
		'''

		MinMax = MinMaxScaler()
		x_new = MinMax.fit_transform(x)

		x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.25)
		model = SVC(C = 1.0, kernel = 'rbf', class_weight = {0: 4.15, 1: 1, 2: 4.15})
		model.fit(x_train, y_train)
		y_predict = model.predict(x_test)
		print ('*****************************')
		print (stock_code)
		print ('y_test')
		print (y_test)
		print ('y_predict')
		print (y_predict)











for i in range(0, 100):
	code = '%06d' % i 
	#print (code)
	timing(code, 0)


