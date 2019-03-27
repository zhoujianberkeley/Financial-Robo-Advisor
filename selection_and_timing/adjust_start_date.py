import tushare as ts

Qua_Dict = {1:'-04-27', 2:'-07-17', 3:'-10-17',4:'-04-17'}

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

def Closest_TraDt_2(tp):
    year = int( tp[0: 4] )
    temp_tp = tp[4: ]
    year += 1
    test = str(year) + temp_tp
    date = int(test[-2:])
    while ts.is_holiday(test):
        date += 1
        test = test[:-2]
        test += str(date)
    return test

assert Closest_TraDt_2('2016-07-17') == '2017-07-17'
print('adjsut start date sucessful')

