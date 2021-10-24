import operator
import numpy as np

def Createdata():
    #创建四组二维特征；
    group = np.array([[61,2],[87,5],[1,45],[4,60]]);
    label = ['动作片','动作片','爱情片','爱情片']

    #返回数据和label标记
    return group,label;

def Kmeans(inX,dataset,labels,k):
  
    """
    Kmeans方法，计算点之间的距离
    然后返回所有训练数据中距离测试点最近的label
    """
    DataSize = dataset.shape[0]
    DataDiff = np.tile(inX,(DataSize,1)) - dataset
    DoubleData = DataDiff ** 2
    SumData = DoubleData.sum(axis=1)
    DistanceData = SumData ** 0.5
    sortDistanceIndex = DistanceData.argsort()
    class_result = {}

    #求最靠近测试点的前k个数据
    for i in range(k):
        #获取距离值最小的前k个元素的label
        fin_label = labels[sortDistanceIndex[i]]
        #修改字典下标的值，get获得字典元素的值，初始值设置为0
        class_result[fin_label] = class_result.get(fin_label,0) + 1

    sortedCount = sorted(class_result.items(),key=operator.itemgetter(1),reverse=True)
    return sortedCount[0][0]

if __name__ == '__main__':
    """
    调用函数打印group和label标签
    """
    group,label = Createdata()
    test = [1,510]
    k = 3

    test_class = Kmeans(test,group,label,k)
    print(test_class)
