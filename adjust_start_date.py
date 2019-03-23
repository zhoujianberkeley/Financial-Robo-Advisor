import tushare as ts

def Closest_TraDt(year, quaer_num):
    '''
    used to test whether a specifica date is a trading date in Chinese A borad
    date: string
    return closest  trading date string
'''
    if quaer_num == 4:
        year += 1

    test = str(year) + Qua_Dict[quaer_num]
    date = int(test[-2:])
    while ts.is_holiday(test):
        date += 1
        test = test[:-2]
        test += str(date)
    return test

print(Closest_TraDt(2016,4))

