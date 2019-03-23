import tushare as ts 
import pandas as pd 
import numpy as np

from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

Model_Rf = RandomForestClassifier()

test_less_row = 100


pd.set_option('display.max_columns', None)

year_list = range(2017, 2018)
quarter_list = range(1, 2)
start_list = ['2017-04-17', '2017-07-17', '2017-10-17', '2018-01-31']
end_list = ['2018-04-17', '2018-07-17', '2018-10-17', '2019-01-31']

temp_row = -1
temp_col = -1

x = []
y = []
x_temp = []


Qua_Dict = {1:'-04-27', 2:'-07-17', 3:'-10-17',4:'-04-17'}


for year in year_list:
	for quarter in quarter_list:
		print (year, quarter)
		df = ts.get_profit_data(year, quarter)
		print ( df.shape )
		print ( df )
		df.dropna( axis=0, how='any', thresh=None, subset=None, inplace=True )
		print ( df )
		df.sort_values( by="code" , ascending=True, inplace=True )
		print ( df )
		df.drop_duplicates( ['code'], inplace = True )
		print ( df )
		df['yield'] = np.nan
		print ( df )
		( temp_row, temp_col ) = df.shape
		print (temp_row, temp_col)
		temp_row = test_less_row ##################################################################
		for row in range(0, temp_row):
			print (row)
			temp_stock = df.iloc[ row, 0 ]
			temp_df = ts.get_hist_data( temp_stock, start = start_list[quarter], end = end_list[quarter] )
			#print (temp_df)
			temp_row2 = -1
			temp_col2 = -1
			( temp_row2, temp_col2 ) = temp_df.shape
			if temp_row2 > 100:
				yield_yoy = ( temp_df.iloc[ 0, 2 ] - temp_df.iloc[ -1, 2 ] ) / temp_df.iloc[ -1, 2 ]
				df.iloc[ row, -1 ] = yield_yoy

		df.dropna( axis=0, how='any', thresh=None, subset=None, inplace=True )
		( temp_row, temp_col ) = df.shape
		for i in range(0, temp_row):
			for j in range(2, temp_col-1):
				x_temp.append(df.iloc[i, j])
			x.append( x_temp )
			y.append( df.iloc[i, -1] )
			x_temp = []

print ('x=')
print (x)
print ('y=')
print (y)

MinMax = MinMaxScaler()
x_new = MinMax.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size=0.25)
print ('y_train =')
print (y_train)
print ('y_test =')
print (y_test)



cv = StratifiedKFold(n_splits= 5, shuffle= True)
C = np.arange(0.5, 5, 0.5)
param_grid = dict(C = C)
kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 7)

model = SVR()
grid_search = GridSearchCV(model, param_grid, scoring = 'neg_mean_squared_error', n_jobs = -1, cv = kfold)
y_train = np.zeros(len(y_train))
grid_result = grid_search.fit(x_train, y_train)
print("Best: %f using %s" % (grid_result.best_score_,grid_search.best_params_))

'''
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print (y_test)
print (y_predict)
'''

###################################################################
row_choosing_stock = -1
col_choosing_stock = -1
def choosing_stock(the_year, the_quarter):
	x_choosing_stock = []
	x_temp_choosing_stock = []
	print (the_year, the_quarter)
	data_choosing_stock = ts.get_profit_data(the_year, the_quarter)
	data_choosing_stock.dropna( axis=0, how='any', thresh=None, subset=None, inplace=True )
	data_choosing_stock.sort_values( by="code" , ascending=True, inplace=True )
	data_choosing_stock.drop_duplicates( ['code'], inplace = True )
	data_choosing_stock['yield'] = np.nan
	( row_choosing_stock, col_choosing_stock ) = data_choosing_stock.shape
	row_choosing_stock = test_less_row ######################################################
	for the_row in range(0, row_choosing_stock):
		for the_col in range(2, col_choosing_stock-1):
			x_temp_choosing_stock.append(data_choosing_stock.iloc[the_row, the_col])
		x_choosing_stock.append(x_temp_choosing_stock)
		x_choosing_stock_new = MinMax.transform(x_choosing_stock)
		y_choosing_stock_predict = model.predict(x_choosing_stock_new)
		x_choosing_stock = []
		x_temp_choosing_stock = []
		data_choosing_stock.iloc[the_row, -1] = y_choosing_stock_predict

	data_choosing_stock.sort_values( by="yield", ascending=False, inplace=True )
	stock_list = data_choosing_stock[['code', 'name', 'yield']]

	print ('stock_list')
	print (stock_list)

##################################################################

choosing_stock(2018, 1)









