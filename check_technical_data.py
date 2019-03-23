import tushare as ts 
import pandas as pd 

df = ts.get_hist_data('000001')

print ( ts.is_holiday('2019-01-31') )