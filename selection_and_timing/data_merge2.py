import pandas as pd 
import tushare as ts 
from adjust_start_date import Closest_TraDt

get_data = 1

row = -1
col = -1

def get_temp_data(year, quarter):
	df1 = ts.get_report_data(year, quarter)
	print (1,"\n", df1)
	df1 = df1.merge(ts.get_profit_data(year, quarter), how = 'inner', on = ['code', 'name'])
	print (2,"\n", df1)
	df1 = df1.merge(ts.get_operation_data(year, quarter), how = 'inner', on = ['code', 'name'])
	print (3, "n", df1)
	df1 = df1.merge(ts.get_growth_data(year, quarter), how = 'inner', on = ['code', 'name'])
	print (4)
	print (df1)
	df1 = df1.merge(ts.get_debtpaying_data(year, quarter), how = 'inner', on = ['code', 'name'])
	print (5)
	print (df1)
	df1 = df1.merge(ts.get_cashflow_data(year, quarter), how = 'inner', on = ['code', 'name'])
	print (6)
	print (df1)
	( row, col ) = df1.shape
	for i in range(0, row):
		df1.iloc[i, 0] = str( df1.iloc[i, 0] )
	return df1

if get_data:
	count = 0
	for i in range(2018, 2019):
		for j in range(3, 4):
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
	df2.drop_duplicates(subset=['code', 'name'], keep='first', inplace=True)
	df2.sort_values( by="code" , ascending=True, inplace=True )
	rows = 0
	cols = 0
	( rows, cols ) = df2.shape
	for j in range(0, rows):
		df2.iloc[j, 0] = str( df2.iloc[j, 0] )
		df2.iloc[j, 0] = df2.iloc[j, 0].zfill(6)
	df2.drop(['net_profits_x', 'profits_yoy', 'quickratio', 'net_profit_ratio', 'bvps', 'epcf', 'distrib', 'report_date', 'roe_y', 'net_profits_y', 'eps_y', 'business_income', 'arturnover', 'arturndays', 'inventory_turnover', 'currentasset_turnover', 'mbrg', 'nprg', 'nav', 'targ', 'epsg', 'seg', 'currentratio', 'cashratio', 'icratio', 'sheqratio', 'adratio', 'cf_sales', 'rateofreturn', 'cf_nm', 'cf_liabilities', 'cashflowratio', 'bips', 'inventory_days', 'currentasset_days', 'gross_profit_rate'], axis=1, inplace=True)
	df2.to_csv('stock_data_new.csv', encoding = 'utf-8')
	( rows, cols ) = df2.shape
	stock_list = []
	for k in range(0, rows):
		if ( (df2.iloc[k, 3] > 20) and (df2.iloc[k, 4] > 15) ):
			stock_list.append(df2.iloc[k, 0])
	print (stock_list)


