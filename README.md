# 机器学习

机器学习指计算机通过观察环境，与环境交互，在吸取信息中学习、自我更新和进步。

对计算机而言，"经验"通常以"数据"形式存在因此机器学习所研究的主要内容，是关于在计算机上从数据中产生"模型"的算法，即"学习算法"。我们把经验数据提供给它，它就能基于这些数据产生模型;在面对新的情况时，模型会给我们提供相应的判断。

简单地说，大多数机器学习算法可以分成 **训练(training)** 和 **测试(testing)** 两个步骤。训练，一般需要训练数据，就是告诉机器前人的经验，比如什么是猫、什么是狗、看到什么该停车。
训练学习的结果，可以认为是机器写的程序或者存储的数据，即模型。

总体上来说，训练包括有监督（supervised learning)和无监督（unsupervised learning)两类。
>有监督好比有老师告诉你正确答案；无监督仅靠观察自学，机器自己在数据里找模式和特征。

## 基本概念

**数据（Data）**：关于研究对象的记录或信息。根据是否包含标签，可分为有标签数据（监督学习）和无标签数据（无监督学习）。

**数据集（Data Set）**：由多个数据记录组成的集合，用于训练或测试机器学习模型。

**样本（Sample）**：数据集中的一条记录，用于训练或测试模型。

**特征（Feature）**：反映事件或对象在某方面的表现或性质的事项，用于训练机器学习模型。

假如我们把某件事物的属性作为坐标轴，例如书本的颜色、类型、页数，则它们张成一个用于描述书本的n维空间，每本书都可在这个空间中找到自己的坐标位置。
由于空间中的每个点对应一个坐标向量，因此我们也把一个样本称为一个 **特征向量(feature vector)**。

**标签（Label）**：关于样本结果的信息，用于监督学习中指导模型学习。

从数据中学得模型的过程称为 **学习 (learning)** 或 **训练 (training)** ，这个过程通过执行某个学习算法来完成。训练过程中使用的数据称为**训练数据(training data)**，
其中每个样本称为一个**训练样本(training sample)**, 训练样本组成的集合称为**训练集 (training set)**。学得模型对应了关于数据的某种潜在的规律。
若我们欲预测的是离散值，例如书的好坏，此类学习任务称为**分类(classification)**; 若欲预测的是连续值，例如书籍受欢迎度：0.85、 0.47。此类学习任务称为**回归(regression)**。

## 方法

模型假设、评价函数和优化算法是构成模型的三个关键要素。

模型假设：我们可以把学习过程看作一个在所有假设组成的空间中进行搜索的过程，搜索目标是找到与训练集匹配的假设，即能够将训练集中的对象判断正确的假设。假设的表示一旦确定，假设空间及其规模大小就确定了。

评价函数：寻找最优之前，我们需要先定义什么是最优，即评价关系的好坏的指标。通常衡量该关系是否能很好的拟合现有观测样本，将拟合的误差最小作为优化目标。

优化算法：设置了评价指标后，就可以在假设圈定的范围内，将使得评价指标最优的关系找出来，这个寻找最优解的方法即为优化算法。

## 常见模型

>机器学习会涉及到许多模型，例如线性模型、决策树、支持向量机、神经网络等等，这里简单介绍线性模型和神经网络

> ### 线性回归
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#引入数据
path =  'data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#看下数据长什么样子
plt.show()
#代价函数
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))
#让我们在训练集中添加一列，以便我们可以使用向量化的解决方案来计算代价和梯度。
data.insert(0, 'Ones', 1)
#现在我们来做一些变量初始化。
# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]#X是所有行，去掉最后一列
y = data.iloc[:,cols-1:cols]#X是所有行，最后一列
#代价函数是应该是numpy矩阵，所以我们需要转换X和Y，然后才能使用它们。 我们还需要初始化theta。
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))
#计算代价函数 (theta初始值为0).
computeCost(X, y, theta)

#批量梯度下降
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0, j] = theta[0, j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost
#初始化一些附加变量 - 学习速率α和要执行的迭代次数。
alpha = 0.01
iters = 1000
#现在让我们运行梯度下降算法来将我们的参数θ适合于训练集。
g, cost = gradientDescent(X, y, theta, alpha, iters)
g
#最后，我们可以使用我们拟合的参数计算训练模型的代价函数（误差）。
computeCost(X, y, g)

#现在我们来绘制线性模型以及数据，直观地看出它的拟合。
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
```




## 线性神经网络

神经元接收到来自n个其他神经元传递过来的输入信号，这些输入信号通过带权重的连接进行传递，
神经元接收到的总输入值将与神经元的阀值进行比较，然后通过"激活函数" (activation function) 处理以产生神经元的输出。



### 感知器

为了理解神经网络，我们应该先理解神经网络的组成单元——**神经元**。神经元也叫做**感知器**。

![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-2.png)

神经元：神经网络中每个节点称为神经元，由两部分组成：

加权和：将所有输入加权求和。

激活函数：加权和的结果经过一个非线性函数变换，让神经元计算具备非线性的能力。



(1) **输入权值** 一个感知器可以接收多个输入：

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-4.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-4.png)

每个输入上有一个**权值**：

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-5.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-5.png)

此外还有一个**偏置项**：

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-6.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-6.png)

就是上图中的w0。

(2) **激活函数** 感知器的激活函数可以有很多选择，比如我们可以选择下面这个**阶跃函数f**来作为激活函数：

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-7.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-7.png)

(3) **输出** 感知器的输出由下面这个公式来计算：

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-8-1.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-8-1.png)

如果看完上面的公式一下子就晕了，不要紧，我们用一个简单的例子来帮助理解。

### 训练一个与函数

| x1   | x2   | y    |
| ---- | ---- | ---- |
| 0    | 0    | 0    |
| 0    | 1    | 0    |
| 1    | 0    | 0    |
| 1    | 1    | 1    |

我们令：

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-10.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-10.png)

而激活函数就f是前面写出来的**阶跃函数**，这时，感知器就相当于**and**函数。不明白？我们验算一下：

输入上面真值表的第一行，即x1=0；x2=0，那么根据公式(1)，计算输出：

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-11.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-11.png)

也就是当x1x2都为0的时候，y为0，这就是**真值表**的第一行。

**获得权重项和偏置项**

感知器训练算法：将权重项和偏置项初始化为0，然后，利用下面的**感知器规则**迭代的修改wi和b，直到训练完成。

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-16.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-16.png)

其中:

[![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-17.png)](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-17.png)

wi是与输入xi对应的权重项，b是偏置项。事实上，可以把b看作是值永远为1的输入xb所对应的权重。t是训练样本的**实际值**，一般称之为**label**。而y是感知器的输出值，它是根据**公式(1)**计算得出。η是一个称为**学习速率**的常数，其作用是控制每一步调整权的幅度。

每次从训练数据中取出一个样本的输入向量x，使用感知器计算其输出y，再根据上面的规则来调整权重。每处理一个样本就调整一次权重。经过多轮迭代后（即全部的训练数据被反复处理多轮），就可以训练出感知器的权重，使之实现目标函数。

```python
# 感知器训练学习
from functools import reduce


class Perceptron:
    def __init__(self, input_num, activator):
        """
        初始化感知器，设置输入参数的个数，以及激活函数。
        激活函数的类型为double -> double
        :param input_num: 输入参数的个数
        :param activator: 激活函数，接受一个double类型的输入，返回一个double类型的输出
        """
        self.activator = activator
        # 权重向量初始化为0
        # 用来存储感知器模型的权重参数。权重参数决定了每个输入特征对感知器输出的影响程度。
        # [0.0 for _ in range(input_num)] 这部分代码使用了列表推导式来创建一个包含
        # input_num 个元素的列表，并将每个元素初始化为0.0。这就构成了初始的权重向量
        self.weights = [0.0 for _ in range(input_num)]
        # 偏置项初始化为0
        self.bias = 0.0

    def __str__(self):
        """
        打印学习到的权重、偏置项
        :return: 学习到的权重和偏置项的字符串表示
        """
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        """
        输入向量，输出感知器的计算结果
        :param input_vec: 输入向量
        :return: 感知器的计算结果
        """
        # 把input_vec[x1,x2,x3...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用map函数计算[x1*w1, x2*w2, x3*w3]
        # 最后利用reduce求和
        return self.activator(
            reduce(lambda a, b: a + b, list(map(lambda x, w: x * w, input_vec, self.weights)), 0.0) + self.bias)

    # iteration 是指训练迭代的次数
    def train(self, input_vecs, labels, iteration, rate):
        """
        输入训练数据：一组向量、与每个向量对应的label；以及训练轮数、学习率
        :param input_vecs: 输入向量的列表
        :param labels: 对应的标签列表
        :param iteration: 训练轮数
        :param rate: 学习率
        """
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):
        """
        一次迭代，把所有的训练数据过一遍
        :param input_vecs: 输入向量的列表
        :param labels: 对应的标签列表
        :param rate: 学习率
        """
        # 把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
        # 而每个训练样本是(input_vec, label)
        samples = zip(input_vecs, labels)
        # 对每个样本，按照感知器规则更新权重
        for (input_vec, label) in samples:
            # 计算感知器在当前权重下的输出
            output = self.predict(input_vec)
            # 更新权重
            self._update_weights(input_vec, output, label, rate)

    def _update_weights(self, input_vec, output, label, rate):
        """
        按照感知器规则更新权重
        :param input_vec: 输入向量
        :param output: 感知器的输出
        :param label: 实际标签
        :param rate: 学习率
        """
        # 把input_vec[x1,x2,x3,...]和weights[w1,w2,w3,...]打包在一起
        # 变成[(x1,w1),(x2,w2),(x3,w3),...]
        # 然后利用感知器规则更新权重
        delta = label - output
        # 新权重 = 旧权重 + 学习率 * 误差 * 输入特征
        self.weights = list(map(lambda x, w: w + rate * delta * x, input_vec, self.weights))
        # 新偏置项 = 旧偏置项 + 学习率 * 误差
        self.bias += rate * delta
        print("权重更新")
        print(self.weights)
        print(self.bias)


def f(x):
    """
    定义激活函数f
    :param x: 输入值
    :return: 激活函数的输出值
    """
    return 1 if x > 0 else 0


def get_training_dataset():
    """
    基于and真值表构建训练数据
    :return: 输入向量列表和对应的标签列表
    """
    # 构建训练数据
    # 输入向量列表
    input_vecs = [[1, 1], [0, 0], [1, 0], [0, 1]]
    # 期望的输出列表，注意要与输入一一对应
    # [1,1] -> 1, [0,0] -> 0, [1,0] -> 0, [0,1] -> 0
    labels = [1, 0, 0, 0]
    return input_vecs, labels


def train_and_perceptron():
    """
    使用and真值表训练感知器
    :return: 训练好的感知器对象
    """
    # 创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    p = Perceptron(2, f)
    # 训练，迭代10轮, 学习速率为0.1
    input_vecs, labels = get_training_dataset()
    p.train(input_vecs, labels, 10, 0.1)
    # 返回训练好的感知器
    return p


if __name__ == '__main__':
    # 训练and感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print('1 and 1 = %d' % and_perception.predict([1, 1]))
    print('0 and 0 = %d' % and_perception.predict([0, 0]))
    print('1 and 0 = %d' % and_perception.predict([1, 0]))
    print('0 and 1 = %d' % and_perception.predict([0, 1]))
```

> 你可能已经看晕了，但是没办法，我们必须理解好这个才能往下走。

**感知器**可以被看作是机器学习中的一种基础模型，特别是在二分类问题中，它可以用于判断一个样本属于哪个类别。虽然感知器相对简单，但它展示了机器学习的几个重要概念和步骤:

1. 数据准备：与其他机器学习模型一样，使用感知器前需要准备训练数据。数据应该包括输入特征和对应的标签或类别，以便模型能够学习输入特征与输出之间的关系。
2. 特征权重和偏置项初始化：感知器的核心是为每个输入特征分配一个权重，并设置一个偏置项。初始时，可以随机初始化这些权重和偏置项。
3. 预测输出：对于给定的输入特征，感知器将计算加权和，并通过阈值函数（如阶跃函数）将结果映射到预测输出。这个预测输出可以被视为二分类的预测结果。
4. 计算损失：将感知器的预测输出与真实标签进行比较，计算损失或误差。常用的损失函数是均方误差（Mean Squared Error）或交叉熵损失（Cross Entropy Loss），具体选择取决于任务类型。
5. 参数更新：使用优化算法（如梯度下降），根据损失函数的梯度来调整感知器的权重和偏置项，以减小损失。这个过程被称为参数更新或模型训练。
6. 重复训练：重复进行步骤3到5，直到达到停止条件。停止条件可以是达到一定的训练轮数、达到一定的精度要求或损失函数收敛等。
7. 预测和评估：训练完成后，使用感知器对新样本进行预测，并评估模型的性能。常用的评估指标包括准确率、精确率、召回率和F1分数等。

> 这是与函数的图形化表示，我们通过跃阶函数将这条线的上方区域转化成1,下方转化称成0。
>
> ![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-14.png)
>
> 但是通过这个感知器，你无法实现以下的函数，异或函数（相同得0,不同得1），因为你找不到一条直线可以把圆形和叉分在两边。
>
> ![深度学习实战教程(一)：感知器](https://cuijiahua.com/wp-content/uploads/2018/10/dl-7-15.png)
>
