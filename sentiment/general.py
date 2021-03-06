import sentiment.predict
import sentiment.load

#封装好的文件，请调用

#新闻来源如下：
source_dic = {'新浪' : 'sina',
              '华尔街财经': 'wallstreetcn',
             '东方财富' : 'eastmoney',
            '云财经' : 'yuncaijing',
            '同花顺' : '10jqka'}
#把训练好的模型加载出来
sentiment.load.loadModel()

#画出一段时间内的情绪指数曲线
def plot(source_list, start, period):
    #换成相应新闻网站的代码，再包装成list
    source = [source_dic[source_list[0]]]

    sentiment.predict.plot_sentiment(source, start, period)
    return


#计算某日的每条新闻的情绪
def cal(data):
    sentiments = sentiment.predict.cal_daily_sentiment(data['content'])
    data['sentiment'] = sentiments
    return data

#提取某日的新闻数据
def extract(source, year, month, date):
    source = source_dic[source]
    news_data = sentiment.predict.load_daily_news(source, year, month, date)
    data = cal(news_data)
    data.index = range(1, data.shape[0] + 1)
    return data

def select_news(i, data):
    return data.loc[i - 1, 'content']



#测试用： uncomment下面的代码， 如果要跑的话
# source = ['同花顺']
# plot(source, '20190204', 20)


# # #选取了一个好看的数据
apr16 = extract('同花顺',2019,4,10).sort_values(by='sentiment',ascending=False).iloc[:10,:]
apr16.to_excel('output/同花顺.xlsx')
# apr16.index = range(0, apr16.shape[0])
# # apr16 = cal(apr16)
# # print(apr16)
# # print(apr16.loc[0,'content'])