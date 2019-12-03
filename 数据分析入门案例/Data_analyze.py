'''
@version: python3.6
@author: Administrator
@file: Data_analyze.py
@time: 2019/11/05
'''
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt
from pylab import mpl
from pandas.io.formats import console, format as fmt
from pandas._config import get_option
from pandas.io.formats.printing import pprint_thing
from scipy.stats import norm, skew
from os import path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
from os import path

import requests
import re
import sys

mpl.rcParams['font.sans-serif'] = ['SimHei'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题
def _put_str(s, space):
    return "{s}".format(s=s)[:space].ljust(space)
def infor(file, verbose=None, buf=None, max_cols=None, null_counts=None):


    lines = []

    lines.append(str(type(file)))
    lines.append(file.index._summary())

    if len(file.columns) == 0:
        lines.append("Empty {name}".format(name=type(file).__name__))
        fmt.buffer_put_lines(buf, lines)
        return

    cols = file.columns

    # hack
    if max_cols is None:
        max_cols = get_option("display.max_info_columns", len(file.columns) + 1)

    max_rows = get_option("display.max_info_rows", len(file) + 1)

    if null_counts is None:
        show_counts = (len(file.columns) <= max_cols) and (len(file) < max_rows)
    else:
        show_counts = null_counts
    exceeds_info_cols = len(file.columns) > max_cols

    def _verbose_repr():
        lines.append("Data columns (total %d columns):" % len(file.columns))
        space = max(len(pprint_thing(k)) for k in file.columns) + 4
        counts = None

        tmpl = "{count}{dtype}"
        if show_counts:
            counts = file.count()
            if len(cols) != len(counts):  # pragma: no cover
                raise AssertionError(
                    "Columns must equal counts "
                    "({cols:d} != {counts:d})".format(
                        cols=len(cols), counts=len(counts)
                    )
                )
            tmpl = "{count} non-null {dtype}"

        dtypes = file.dtypes
        for i, col in enumerate(file.columns):
            dtype = dtypes.iloc[i]
            col = pprint_thing(col)

            count = ""
            if show_counts:
                count = counts.iloc[i]

            lines.append(
                _put_str(col, space) + tmpl.format(count=count, dtype=dtype)
            )

    def _non_verbose_repr():
        lines.append(file.columns._summary(name="Columns"))


    if verbose:
        _verbose_repr()
    elif verbose is False:  # specifically set to False, not nesc None
        _non_verbose_repr()
    else:
        if exceeds_info_cols:
            _non_verbose_repr()
        else:
            _verbose_repr()

    counts = file._data.get_dtype_counts()
    dtypes = ["{k}({kk:d})".format(k=k[0], kk=k[1]) for k in sorted(counts.items())]
    lines.append("dtypes: {types}".format(types=", ".join(dtypes)))


    # fmt.buffer_put_lines(buf, lines)
    if any(isinstance(x, str) for x in lines):
        lines = [str(x) for x in lines]
    return "\n".join(lines)



class DataAnalyze:
    def __init__(self,file):
        self.file=file
        self.df=pd.read_csv(file)

    #返回数据的形状和每列数据的属性（包括索引范围、列属性的数量，是否为空，数据类型等）
    def check_structure(self):
        df = self.df
        shape='行： '+str(len(df.index))+'列： '+str(len(df.columns))+'\n'

        info='数据列信息\n'+infor(df)
        return shape+info

    #用热力图来描述数值型特征之间的相关性
    def correlation(self):
        df = self.df
        corrmat = df.corr()#两两之间关系的数值表示。值越高，关系越密切
        print(corrmat)
        # sns.pairplot(df)
        sns.heatmap(df.corr(), cmap="YlGnBu")
        plt.title('数值型数据关系热力图')
        plt.legend()
        # plt.savefig("picture/heatmap.jpg")
        plt.show()

    #用对角线图描述数值型特征的之间的分布（网格图中包含柱状分布图，散点分布图和密度图）
    def dig_dist(self):
        df = self.df
        g = sns.PairGrid(df)  # 主对角线是数据集中每一个数值型数据列的直方图，
        g.map_diag(sns.distplot)  # 指定对角线绘图类型
        g.map_upper(plt.scatter, edgecolor="white")  # 两两关系分布
        g.map_lower(sns.kdeplot)  # 绘制核密度分布图。
        plt.title('数值型数据分布图')
        plt.legend()
        # plt.savefig("picture/distplot.jpg")
        plt.show()

    #返回缺省值的统计情况（包括饼状图和水平柱状图），并且补全缺省值。最后返回缺省的统计数据以及缺省填充后的新的全体数据
    def default_value(self):
        df = self.df
        #统计数量，比率
        count = df.isnull().sum().sort_values(ascending=False)#统计出每个属性的缺省数量，再根据缺省数量倒排序
        ratio = count / len(df)#每个属性的缺省占自己总数的比率
        nulldata = pd.concat([count, ratio], axis=1, keys=['count', 'ratio'])

        #饼状图
        explode = [0]
        explode = explode * len(nulldata.index)
        explode[0] = 0.1
        plt.pie(x=count, labels=nulldata.index, autopct='%1.1f%%', shadow=True, startangle=90,
                explode=explode, pctdistance=0.8, textprops={'fontsize': 16, 'color': 'w'})#饼状图画出每个属性的缺省在整体缺省数据的占比
        plt.title('属性缺省占比图')
        plt.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), borderaxespad=0.3)
        # plt.savefig("picture/default_value_pie.jpg")
        plt.show()

        #水平柱状图
        plt.figure()
        plt.barh(nulldata.index, count, 0.5, color='#FFFF00', label="缺省")#每个属性的缺省样本数
        plt.barh(nulldata.index, df.shape[0] - count, 0.5, color='#97FFFF', left=count, label="不缺省")#每个属性的样本值不缺省的数量
        plt.xlabel("属性")
        plt.ylabel("样本数")
        plt.legend()
        plt.title("每个属性的缺省情况")
        # plt.savefig("picture/default_value_bar.jpg")
        plt.show()

        # 填充缺省，字符串类型的属性用None填充,数值型用众数填充。简单填充，可能会产生更多的误差
        for index in nulldata.index:
            if type(index) is not object:
                df[index].fillna("None", inplace=True)
            else:
                df[index].fillna(df[index].mode()[0], inplace=True)
        self.df=df
        return nulldata,df


    #将偏值较大的log归正: 用柱状图将数值型属性的偏值表示出，最终会返回它们的偏值和进行log1p后新的数据
    def Skew(self):
        df=self.df
        skew_value=np.abs(df.skew()).sort_values(ascending=False)
        skew_value=skew_value[skew_value>0.5]

        #用柱状图描述各个属性的偏值
        sns.barplot(skew_value.index, skew_value, palette="BuPu_r", label="偏值")
        plt.title('数值型属性的偏值')
        plt.xlabel('属性')
        plt.ylabel('偏值skew')
        plt.legend()
        # plt.savefig("picture/skew_pie.jpg")
        plt.show()

        #对于偏值大于0.15的定量进行正态化
        X_numeric = df.select_dtypes(exclude=["object"])
        skewness = X_numeric.apply(lambda x: skew(x))
        skewness_features = skewness[abs(skewness) >= 0.5].index
        df[skewness_features] = np.log1p(df[skewness_features])
        self.df=df

        return skew_value,df

    #定义生成词云的方法
    def generate_wordcloud(self,text,name, mask='Images/man_mask.png'):
        '''
        输入文本生成词云,如果是中文文本需要先进行分词处理
        '''

        # 设置显示方式
        d = path.dirname(__file__)
        alice_mask = np.array(Image.open(path.join(d, mask)))
        font_path = path.join(d, "font//msyh.ttf")
        stopwords = set(STOPWORDS)
        stopwords.add('|')
        wc = WordCloud(background_color="white",  # 设置背景颜色
                       max_words=2000,  # 词云显示的最大词数
                       mask=alice_mask,  # 设置背景图片
                       stopwords=stopwords,  # 设置停用词
                       font_path=font_path,  # 兼容中文字体，不然中文会显示乱码
                       )

        # 生成词云
        wc.generate(text)

        # # 生成的词云图像保存到本地
        # wc.to_file("picture/"+name+"词云.png")
        plt.figure()
        # 显示图像
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")  # 关掉图像的坐标
        plt.show()
    #调用生成词云的方法，给每个非数值型特征生成词云保存在本地
    def generate_img(self):
        df=self.df
        for col in df.columns:
            if df[col].dtype == object:
                text = '|'.join(df[col].tolist())
                print(text)
                self.generate_wordcloud(text, col)

data=DataAnalyze('data/drinks.csv')
# print(data.check_structure())
# data.correlation()
# data.dig_dist()
null,df=data.default_value()
# skew,df=data.Skew()
data.generate_img()





