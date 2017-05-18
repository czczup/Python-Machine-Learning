import matplotlib.pyplot as plt
#加载matplotlib用于数据的可视化
from sklearn.decomposition import PCA
#加载PCA主成分分析算法包
from sklearn.datasets import load_iris
#加载鸢尾花数据集导入函数

data = load_iris() #以字典形式加载鸢尾花数据集
y = data.target #使用y表示数据集中的标签
x = data.data #使用x表示数据集中的属性数据
pca = PCA(n_components=2) #加载PCA算法，设置将为后主成分数目为2
reduced_X = pca.fit_transform(x) #对原始数据进行降维，保存在reduced_X中

red_x,red_y = [],[]
#第一类数据点
blue_x,blue_y = [],[]
#第二类数据点
green_x,green_y = [],[]
#第三类数据点

for i in range(len(reduced_X)):
    if y[i] == 0:
        red_x.append(reduced_X[i][0])
        red_y.append(reduced_X[i][1])
    elif y[i] == 1:
        blue_x.append(reduced_X[i][0])
        blue_y.append(reduced_X[i][1])
    else:
        green_x.append(reduced_X[i][0])
        green_y.append(reduced_X[i][1])

plt.scatter(red_x,red_y,c='r',marker='x') #第一类数据点
plt.scatter(blue_x,blue_y,c='b',marker='D') #第二类数据点
plt.scatter(green_x,green_y,c='g',marker='.') #第三类数据点
plt.show()
#可视化