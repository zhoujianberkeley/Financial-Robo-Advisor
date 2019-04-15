# encoding utf-8
from os import listdir
from os.path import isfile, join
import jieba
import codecs
from sentiment.langconv import * # convert Traditional Chinese characters to Simplified Chinese characters
import pickle
import random
import numpy as np
import scipy.stats as stats
import pylab as pl
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
pro = ts.pro_api('a267cd47f74302b6be5d5d5a1ccbeba29b92620482619bd108a03ee2')

# load model
model = None
sentiment_tag = None
maxLength = None

def loadModel():
    global model, sentiment_tag, maxLength
    metaData = sentiment.load.__loadStuff("./data/meta_sentiment_chinese.p")
    maxLength = metaData.get("maxLength")
    vocab_size = metaData.get("vocab_size")
    output_dimen = metaData.get("output_dimen")
    sentiment_tag = metaData.get("sentiment_tag")
    embedding_dim = 256
    if model is None:
        model = Sequential()
        model.add(Embedding(vocab_size, embedding_dim, input_length=maxLength))
        # Each input would have a size of (maxLength x 256) and each of these 256 sized vectors are fed into the GRU layer one at a time.
        # All the intermediate outputs are collected and then passed on to the second GRU layer.
        model.add(GRU(256, dropout=0.9, return_sequences=True))
        # Using the intermediate outputs, we pass them to another GRU layer and collect the final output only this time
        model.add(GRU(256, dropout=0.9))
        # The output is then sent to a fully connected layer that would give us our final output_dim classes
        model.add(Dense(output_dimen, activation='softmax'))
        # We use the adam optimizer instead of standard SGD since it converges much faster
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.load_weights('./data/sentiment_chinese_model.HDF5')
        model.summary()
    print("Model weights loaded!")


def findFeatures(text):
    text = Converter('zh-hans').convert(text)
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    seg_list = jieba.cut(text, cut_all=False)
    seg_list = list(seg_list)
    text = " ".join(seg_list)
    textArray = [text]
    input_tokenizer_load = sentiment.load.__loadStuff("./data/input_tokenizer_chinese.p")
    textArray = np.array(pad_sequences(input_tokenizer_load.texts_to_sequences(textArray), maxlen=maxLength))
    return textArray


def predictResult(text):
    if model is None:
        print("Please run \"loadModel\" first.")
        return None
    features = findFeatures(text)
    predicted = model.predict(features)[0]  # we have only one sentence to predict, so take index 0
    predicted = np.array(predicted)
    probab = predicted.max()
    predition = sentiment_tag[predicted.argmax()]
    if predition == 'neg':
        probab = -probab
    return probab


# draw sentiment diagram
def num_to_date(year, month, date):
    return datetime.datetime(year, month, date)


def date_to_str(datetime):
    return datetime.strftime('%Y-%m-%d').replace('-', '')


def cal_avg_daily_sentiment(series):
    return np.average(series.apply(lambda x: predictResult(x)))


def cal_daily_sentiment(series):
    return series.apply(lambda x: predictResult(x))


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


