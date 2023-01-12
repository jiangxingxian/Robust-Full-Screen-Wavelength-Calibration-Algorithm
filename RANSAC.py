import numpy as np
import matplotlib.pyplot as plt
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import math
import heapq
from sklearn.linear_model import LinearRegression, TheilSenRegressor




# 数据量。
SIZE = 83
lr = 0.001
beta_huber = np.zeros((2, 1))-0.1
beta_tukey = np.zeros((2, 1))-0.1
sigma1 = 0.001
sigma2 = 0.001
num_epoch = 150

#正弦曲线拟合
def Fittingcurve1(x,y,parameter1,parameter2):
    '''
    :param x: x轴数据
    :param y: y轴数据
    :param parameter1: x的含义
    :param parameter2: y的含义
    :return: 拟合曲线参数para,拟合曲线函数形式target_func
    '''
    # 假设x与y的函数关系
    def target_func(x, a0, a1, a2, a3):
        return a0 * np.sin(a1 * x + a2) + a3

    # 拟合sin曲线
    fs = np.fft.fftfreq(len(x), x[1] - x[0])
    Y = abs(np.fft.fft(y))
    freq = abs(fs[np.argmax(Y[1:]) + 1])
    a0 = max(y) - min(y)
    a1 = 2 * freq
    a2 = 0
    a3 = np.mean(y)
    p0 = [a0, a1, a2, a3]
    para, _ = optimize.curve_fit(target_func, x, y, p0=p0, maxfev=5000000)
    print("------------------"+parameter1+"与"+parameter2+"的正弦拟合曲线--------------------------")
    print(parameter1+"与"+parameter2+"的正弦拟合曲线为：y = " + str(para[0]) + "* sin(" + str(para[1]) + \
          " * x + " + str(para[2]) + ") + " + str(para[3]))
    print("------------------"+parameter1+"与"+parameter2+"的正弦拟合曲线--------------------------")
    return para,target_func

def Fittingcurve1_(x, para_):

    return para_[0] * np.sin(para_[1] * x + para_[2]) + para_[3]

#拟合曲线作图
def showcurve1(x,y,para,target_func,parameter1,parameter2,parameter3):
    '''
    :param x: x轴数据
    :param y: y轴数据
    :param para:
    :param target_func:
    :param parameter1:
    :param parameter2:
    :param parameter3:
    :return:作图
    '''
    # 原始数据作图
    fig, ax = plt.subplots()

    y_fit = [target_func(a, *para) for a in x]
    print("正弦函数——原始数据与拟合数据平均差距", np.max(np.abs(y_fit - y)))
    # 计算出来的数据作图
    ax.plot(x, y_fit, 'k')
    ax.scatter(x, y)
    # ax.set_title('Sinusoidal Curve', fontsize=16, color='black')
    plt.xlabel(parameter2, fontsize=16)
    plt.ylabel(parameter3, fontsize=16)
    plt.legend(['Sinusoidal Curve'], loc='upper left')
    plt.show()
    print("y = " + str(para[0]) + "* sin(" + str(para[1]) + \
                " * x + " + str(para[2]) + ") + " + str(para[3]))

#线性拟合
def Fittingcurve2(x,y,parameter1,parameter2):
    '''
    :param x: x轴数据
    :param y: y轴数据
    :param parameter1: x的含义
    :param parameter2: y的含义
    :return: 拟合曲线参数para
    '''
    polypara = np.polyfit(x, y, deg=1)
    print("------------------"+parameter1+"与"+parameter2+"的线性拟合曲线--------------------------")
    print(parameter1+"与"+parameter2+"的线性拟合曲线为：y = " + str(polypara[0]) + " * x + " + str(polypara[1]))
    print("------------------"+parameter1+"与"+parameter2+"的线性拟合曲线--------------------------")
    return polypara

#拟合曲线作图
def showcurve2(x,y,para,parameter1,parameter2,parameter3):
    '''
    :param x: x轴数据
    :param y: y轴数据
    :param para:
    :param parameter1:
    :param parameter2:
    :param parameter3:
    :return:作图
    '''
    # 原始数据作图
    fig, ax = plt.subplots()

    y_fit = para[0]*x + para[1]
    print("线性函数——原始数据与拟合数据平均差距", np.max(np.abs(y_fit - y)))
    # 计算出来的数据作图
    ax.plot(x, y_fit, 'g')
    ax.scatter(x, y)
    ax.set_title(parameter1+'光栅零点'+parameter2+'与'+parameter3+'的线性拟合曲线', fontsize=12, color='black')
    plt.xlabel(parameter2)
    plt.ylabel(parameter3)
    plt.legend(["线性拟合曲线为：y = " + str(para[0]) + " * x + " + str(para[1]), '原始数据'], loc='upper left')
    plt.show()

def weight_huber(y, y_pred, sigma):
    """
    Huber的权重w
    :param y: 真实值
    :param y_pred: 计算值
    :param sigma: 超参
    :return: w
    """
    if np.abs(y-y_pred) <= sigma:
        w = 1
    else:
        w = sigma/np.abs(y-y_pred)

    return w

def weight_tukey(y, y_pred, sigma):
    """
    Tukey的权重w
    :param y: 真实值
    :param y_pred: 计算值
    :param sigma: 超参
    :return: w
    """
    if (1-np.power((y-y_pred)/sigma, 2)) <= 0:
        w = 0
    else:
        w = np.power((1-np.power((y-y_pred)/sigma, 2)), 2)

    return w


def beta_updata(beta, lr, X, Y, sigma, method):

    f_derivative = np.zeros((2, 1))

    for i in range(X.shape[1]):
        a = X[:, i]
        y_pred = np.dot(X[:, i].T, beta)
        if method == 1:
            w = weight_huber(Y[i], y_pred, sigma)
        else:
            w = weight_tukey(Y[i], y_pred, sigma)
        f_derivative = f_derivative - w*X[:, i].reshape(2, 1)*(Y[i] - y_pred)
    beta = beta - lr*f_derivative

    return beta



def beta_leastsquare(X, Y):

    print(X.shape, Y.shape)
    beta = np.dot(np.dot(np.linalg.inv(np.dot(X, X.T)), X), Y)

    return beta

def plot_fit(X, beta1, beta2, beta3):

    y_fit1 = np.dot(X.T, beta1)
    plt.plot(X[0, :], y_fit1,  'b')
    y_fit2 = np.dot(X.T, beta2)
    plt.plot(X[0, :], y_fit2,  'r')
    y_fit3 = np.dot(X.T, beta3)
    plt.plot(X[0, :], y_fit3,  'g')


def res(X, Y, beta1, beta2, beta3, beta4, y_fit5, outliers_remove=False):
    if outliers_remove is False:
        y_fit1 = np.dot(X.T, beta1)
        y_fit2 = np.dot(X.T, beta2)
        y_fit3 = np.dot(X.T, beta3)
        differ1 = np.abs(Y - y_fit1)
        differ2 = np.abs(Y - y_fit2)
        res1 = np.sum((Y - y_fit1) ** 2)
        res2 = np.sum((Y - y_fit2) ** 2)
        res3 = np.sum((Y - y_fit3) ** 2)
        Len = X.shape[1]
        differ1_max = np.max(differ1)
        differ2_max = np.max(differ2)
    else:

        y_fit1 = np.dot(X.T, beta1)
        y_fit2 = np.dot(X.T, beta2)
        y_fit3 = np.dot(X.T, beta3)
        y_fit4 = np.dot(X.T, beta4)
        differ1 = np.abs(Y - y_fit1)
        differ2 = np.abs(Y - y_fit2)
        differ3 = np.abs(Y - y_fit3)
        differ4 = np.abs(Y - y_fit4)
        differ5 = np.abs(Y - y_fit5)
        index = heapq.nlargest(9, range(len(differ1)), differ1.take)
        print("outliers_remove:", index)
        X = np.delete(X, index, axis=1)
        Y = np.delete(Y, index, axis=0)
        y_fit1 = np.dot(X.T, beta1)
        y_fit2 = np.dot(X.T, beta2)
        y_fit3 = np.dot(X.T, beta3)
        y_fit4 = np.dot(X.T, beta4)
        differ1 = np.abs(Y - y_fit1)
        differ2 = np.abs(Y - y_fit2)
        differ4 = np.abs(Y - y_fit4)
        y_fit5 = np.delete(y_fit5, index, axis=0)
        differ5 = np.abs(Y - y_fit5)
        res1 = np.sum((Y - y_fit1) ** 2)
        res2 = np.sum((Y - y_fit2) ** 2)
        res3 = np.sum((Y - y_fit3) ** 2)
        res4 = np.sum((Y - y_fit4) ** 2)
        res5 = np.sum((Y - y_fit5) ** 2)
        Len = X.shape[1]
        differ1_max = np.max(differ1)
        differ2_max = np.max(differ2)
        differ4_max = np.max(differ4)
        differ5_max = np.max(differ5)

    return res1/Len, res2/Len, res3/Len, res4/Len, res5/Len, Len, np.sum(differ1), np.sum(differ2), np.sum(differ4), np.sum(differ5), differ1_max, differ2_max, differ4_max, differ5_max


def main_ht(X, Y, lr, num_epoch, sigma1, sigma2, beta_huber, beta_tukey, RANSAC_bata, y_fit5):
    plt.scatter(X[0, :], Y)
    plt.xlabel('△p', fontsize=16)
    plt.ylabel('△x', fontsize=16)
    plt.show()
    beta_leastsquares = beta_leastsquare(X, Y)
    for i in range(num_epoch):
        beta_huber = beta_updata(beta_huber, lr, X, Y, sigma1, 1)
        beta_tukey = beta_updata(beta_tukey, lr, X, Y, sigma2, 2)
    beta_huber = np.squeeze(beta_huber)
    beta_tukey = np.squeeze(beta_tukey)

    print("--------------------------参数--------------------------")
    print("beta_leastsquares:", beta_leastsquares)
    print("beta_huber:", beta_huber.squeeze())
    print("beta_tukey:", beta_tukey.squeeze())

    res1, res2, res3, res4, res5, Len, differ1, differ2, differ4, differ5, differ1_max, differ2_max, differ4_max, differ5_max = res(X, Y, beta_leastsquares, beta_huber, beta_tukey, RANSAC_bata, y_fit5, outliers_remove=True)
    print("outliers_remove", "Leastsquares:", res1, "huber:", res2, "tukey:", res3, "RANSAC:", res4, "Theil-Sen:", res5)
    # plt.scatter(1, res2, c='black')
    # plt.scatter(2, res4, c='red')
    # plt.scatter(3, res5, c='blue')
    # plt.scatter(1, differ2_max, c='k')
    # plt.scatter(2, differ4_max, c='c')
    # plt.scatter(3, differ5_max, c='gold')
    # plt.scatter(1, differ2/Len, c='g')
    # plt.scatter(2, differ4/Len, c='y')
    # plt.scatter(3, differ5 / Len, c='yellowgreen')

    plt.bar(np.arange(3), [res1, differ1_max, differ1/Len], width=0.15, color='b', label='LS')
    plt.bar(np.arange(3) + 0.2 * 1, [res4, differ4_max, differ4 / Len], width=0.15, color='#0066CC', label='RANSAC')
    plt.bar(np.arange(3) + 0.2 * 2, [res5, differ5_max, differ5 / Len], width=0.15, color='#339966', label='Theil-Sen')
    plt.bar(np.arange(3) + 0.2 * 3, [res2, differ2_max, differ2 / Len], width=0.15, color='c', label='Huber')
    plt.legend(loc="upper right", prop={"size": 12, })
    plt.xticks(np.arange(3)+0.1, ["MSE", "RES", "MAE"], size=14)
    plt.ylim(1, 160)
    # plt.text(1, res2, "{:.2f}".format(res2), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(2, res4, "{:.2f}".format(res4), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(3, res5, "{:.2f}".format(res5), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(1, differ2_max, "{:.2f}".format(differ2_max), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(2, differ4_max, "{:.2f}".format(differ4_max), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(3, differ5_max, "{:.2f}".format(differ5_max), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(1, differ2/Len, "{:.2f}".format(differ2/Len), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(2, differ4/Len, "{:.2f}".format(differ4/Len), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(3, differ5 / Len, "{:.2f}".format(differ5 / Len), fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    plt.xlabel('Method', fontsize=14)
    # plt.xticks(np.arange(0, 4, 1))
    plt.ylabel('VALUE', fontsize=14)
    # plt.legend(['Huber_MSE', "RANSAC_MSE", "Theil-Sen_MSE", "Huber_RES", "RANSAC_RES", "Theil-Sen_RSE", "Huber_MAE", "RANSAC_MAE", "Theil-Sen_MAE"], loc='upper right')
    plt.show()
    print("--------------------------参数--------------------------")
    return beta_leastsquares, beta_huber, beta_tukey


# 产生数据。np.linspace 返回一个一维数组，SIZE指定数组长度。
# 数组最小值是0，最大值是10。所有元素间隔相等。
delta_y_4 = np.array(
            [-463, -250, -63, 0, 188, 325, 522, -581, -400, -200, 0, 141, 309, 450, -508, -338, -123, 0, 200
                , 400, 506, -494, -228, -119, 0, 181, 376, 496, -495, -321, -172, 0, 205, 311, 512, -455, -319, -185, 0,
             222, 364, 515, -301, -131
                , 0, 166, 332, 482, -454, -284, -154, 0, 146, 347, -418, -246, -115, 0, 194, 385, -471, -323, -160, 0, 234,
             365, 434, -440, -259, -151
                , 0, 202, 347, 487, -272, -362, -168, -258, -85, -90, 90, 0, 278])  # 样机4
delta_p_4 = np.array([871, 472, 119, 0, -344, -600, -959
                                 , 823
                                 , 621
                                 , 245
                                 , 0
                                 , -401
                                 , -716
                                 , -978
                                 , 960
                                 , 637
                                 , 235
                                 , 0
                                 , -375
                                 , -744
                                 , -942
                                 , 934
                                 , 432
                                 , 223
                                 , 0
                                 , -344
                                 , -711
                                 , -935
                                 , 946
                                 , 609
                                 , 324
                                 , 0
                                 , -390
                                 , -593
                                 , -970
                                 , 883
                                 , 618
                                 , 359
                                 , 0
                                 , -423
                                 , -695
                                 , -979
                                 , 586
                                 , 251
                                 , 0
                                 , -324
                                 , -644
                                 , -930
                                 , 890
                                 , 557
                                 , 300
                                 , 0
                                 , -284
                                 , -671
                                 , 824
                                 , 485
                                 , 226
                                 , 0
                                 , -381
                                 , -753
                                 , 954
                                 , 654
                                 , 323
                                 , 0
                                 , -460
                                 , -722
                                 , -987
                                 , 898
                                 , 525
                                 , 305
                                 , 0
                                 , -410
                                 , -702
                                 , -979
                                 , 567
                                 , 758
                                 , 351
                                 , 541
                                 , 176
                                 , 0
                                 , -186
                                 , 0
                                 , -571
                              ])  # 样机4
# delta_p_4 = np.vstack((delta_p_4, np.ones_like(delta_p_4)))




def RANDOM_method(x, y):

    RANDOM_X = np.array(x) # 散点图的横轴。
    RANDOM_Y = np.array(y) # 散点图的纵轴。

    # 使用RANSAC算法估算模型
    # 迭代最大次数，每次得到更好的估计会优化iters的数值
    iters = 100000
    # 数据和模型之间可接受的差值
    sigma = 0.25
    # 最好模型的参数估计和内点数目
    best_a = 0
    best_b = 0
    pretotal = 0
    # 希望的得到正确模型的概率
    P = 0.99
    for i in range(iters):
        # 随机在数据中红选出两个点去求解模型
        sample_index = random.sample(range(SIZE),2)
        x_1 = RANDOM_X[sample_index[0]]
        x_2 = RANDOM_X[sample_index[1]]
        y_1 = RANDOM_Y[sample_index[0]]
        y_2 = RANDOM_Y[sample_index[1]]

        # y = ax + b 求解出a，b
        a = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - a * x_1

        # 算出内点数目
        total_inlier = 0
        for index in range(SIZE):
            y_estimate = a * RANDOM_X[index] + b
            if abs(y_estimate - RANDOM_Y[index]) < sigma:
                total_inlier = total_inlier + 1

        # 判断当前的模型是否比之前估算的模型好
        if total_inlier > pretotal:
            iters = math.log(1 - P) / math.log(1 - pow(total_inlier / (SIZE * 2), 2))
            pretotal = total_inlier
            best_a = a
            best_b = b

        # 判断是否当前模型已经符合超过一半的点
        if total_inlier > SIZE:
            break
    # 用我们得到的最佳估计画图
    Y = best_a * RANDOM_X + best_b
    return [a, b]



RANSAC_bata = RANDOM_method(delta_p_4, delta_y_4)
Theil = TheilSenRegressor(random_state=42)
delta_p_4_ = delta_p_4[:, np.newaxis]
Theil.fit(delta_p_4_, delta_y_4)
y_pred_Theil = Theil.predict(delta_p_4_)
delta_p_4 = np.vstack((delta_p_4, np.ones_like(delta_p_4)))
p = delta_p_4
y = delta_y_4
beta_leastsquares, beta_huber_1200, beta_tukey_1200 = main_ht(p, y, lr, num_epoch, sigma1, sigma2,
                                                              beta_huber, beta_tukey, RANSAC_bata, y_pred_Theil)