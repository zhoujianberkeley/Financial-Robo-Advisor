import sentiment.predict
import sentiment.load

#封装好的文件，请调用

#新闻来源如下：
#source_list = ['sina', 'wallstreetcn', 'eastmoney', 'yuncaijing', '10jqka']
#                  新浪    华尔街财经       东方财富。     云财经。      同花顺


#把训练好的模型加载出来
sentiment.load.loadModel()

#画出一段时间内的情绪指数曲线
def plot(source_list, start, period):
    sentiment.predict.plot_sentiment(source_list, start, period)
    return

source_list = ['sina', 'wallstreetcn']
plot(source_list, '20190404', 10)

#提取某日的新闻数据
def extract(source, year, month, date):
    news_data = sentiment.predict.load_daily_news(source, year, month, date)
    return news_data

#计算某日的每条新闻的情绪
def cal(data):
    sentiments = sentiment.predict.cal_daily_sentiment(data['content'])
    data['sentiment'] = sentiments
    return data

def select_news(i, data):
    return data.loc[i - 1, 'content']


#选取了一个好看的数据
apr16 = extract('sina',2019,4,13).loc[[9,8,11,42,66,50,19,102,68],:]
apr16.index = range(0, apr16.shape[0])
apr16 = cal(apr16)
print(apr16)
print(apr16.loc[0,'content'])