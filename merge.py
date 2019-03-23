import pandas as pd 
import tushare as ts 
from adjust_start_date import Closest_TraDt

def get_temp_data(year, quarter):
	df1 = ts.get_report_data(year, quarter)
	df1 = df1.merge(ts.get_profit_data(year, quarter), how = 'inner', on = ['code', 'name'])
	df1 = df1.merge(ts.get_operation_data(year, quarter), how = 'inner', on = ['code', 'name'])
	df1 = df1.merge(ts.get_growth_data(year, quarter), how = 'inner', on = ['code', 'name'])
	df1 = df1.merge(ts.get_debtpaying_data(year, quarter), how = 'inner', on = ['code', 'name'])
	df1 = df1.merge(ts.get_cashflow_data(year, quarter), how = 'inner', on = ['code', 'name'])
	return df1

count = 0
for i in range(2016, 2019):
	for j in range(1, 5):
		print (i)
		print (j)
		if count == 0:
			df2 = get_temp_data(i, j)
			df2['time_point'] = Closest_TraDt(i, j)
		else:
			df2_temp = get_temp_data(i, j)
			df2_temp['time_point'] = Closest_TraDt(i, j)
			df2 = pd.concat([df2, df2_temp])
			
		count += 1

print (df2)
df2.to_csv('stock_data.csv', encoding = 'utf-8')