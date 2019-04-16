# encoding utf-8
import jieba
from sentiment.langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tushare as ts

import sentiment.load

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import TensorBoard


# 设定调取tushare api的token
pro = ts.pro_api('3e1bc7304a72d89ecaf3ce27d1f15fc04d19da1c62edafb34cbac17d')
# 周健的token 'a267cd47f74302b6be5d5d5a1ccbeba29b92620482619bd108a03ee2'

# load model
model = None
sentiment_tag = None
maxLength = None

# draw sentiment diagram
def num_to_date(year, month, date):
    return datetime.datetime(year, month, date)


def date_to_str(datetime):
    return datetime.strftime('%Y-%m-%d').replace('-', '')


def cal_avg_daily_sentiment(series):
    return np.average(series.apply(lambda x: sentiment.load.predictResult(x)))


def cal_daily_sentiment(series):
    return series.apply(lambda x: sentiment.load.predictResult(x))


def load_news(source, start_date):
    '''
    start_date 必须是datetime格式的
    根据datetime提取新闻
    '''
    end_date = start_date + datetime.timedelta(1)
    df = pro.news(src=source, start_date=date_to_str(start_date), \
                  end_date=date_to_str(end_date))
    return df


def load_daily_news(source, year, month, day):
    '''
    根据year month day提取日新闻数据
    '''
    start_date = num_to_date(year, month, day)
    return load_news(source, start_date)


def load_news_data(source, year, month, start, period, top_n=10):
    '''
    start_date='20190410', end_date='20181122'
    sina	获取新浪财经实时资讯
    华尔街见闻	wallstreetcn
    同花顺	10jqka
    东方财富	eastmoney
    云财经	yuncaijing
    '''
    # 确定开始时间
    start_date = num_to_date(year, month, start)

    sentiment_indices = []
    time_list = []
    for date in range(1, period + 1):
        time_list.append(start_date)
        # 把日期换成string格式
        df = load_news(source, start_date)
        sentiment_index = cal_avg_daily_sentiment(df['content'][:top_n])
        sentiment_indices.append(sentiment_index)
        start_date = start_date + datetime.timedelta(1)
    if len(sentiment_indices) > 1:
        sentiment_indices = sentiment_indices - np.average(sentiment_indices)
    return time_list, sentiment_indices


def visual(time, y_list, y_label):
    for y, label in zip(y_list, y_label):
        plt.plot(time, y, label=label)
    plt.xticks(rotation=45, fontsize=8)
    plt.axhline(y=0, c='black')
    plt.axhline(y=0.3, c='r', linestyle='--')
    plt.axhline(y=-0.3, c='g', linestyle='--')
    plt.legend()
    plt.title('sentiment index')
    plt.xlabel('date')
    plt.ylabel('sentiment indices')
    plt.show()
    plt.close()


def plot_sentiment(source_list, start, period):
    '''
    封装用函数
    source_list = ['sina', 'wallstreetcn','eastmoney', 'yuncaijing','10jqka']
                    新浪.    华尔街财经。      东方财富。     云财经。      同花顺
    start_date = '20171210' 开始的日期，请一定严格按照次格式
    period = '30' 时间持续日期
    '''
    year = int(start[:4])
    month = int(start[4:6])
    start_date = int(start[6:])
    y_list = []
    y_label = []
    for source in source_list:
        print("extracing data from {0}".format(source))
        time, news_data = load_news_data(source, year,
                            month, start = start_date, period = period, top_n = 10)
        y_list.append(news_data)
        y_label.append(source)

    visual(time, y_list, y_label)
    return


