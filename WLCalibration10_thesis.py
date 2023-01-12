import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as optimize
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import math
import heapq

pi = np.pi
plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False



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



def res(X, Y, beta1, beta2, beta3, outliers_remove=False):
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
        differ1 = np.abs(Y - y_fit1)
        differ2 = np.abs(Y - y_fit2)
        differ3 = np.abs(Y - y_fit3)
        index = heapq.nlargest(9, range(len(differ1)), differ1.take)
        print("outliers_remove:", index)
        X = np.delete(X, index, axis=1)
        Y = np.delete(Y, index, axis=0)
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

    return res1/Len, res2/Len, res3/Len, Len, np.sum(differ1), np.sum(differ2), differ1_max, differ2_max


def main_ht(X, Y, lr, num_epoch, sigma1, sigma2, beta_huber, beta_tukey):
    plt.scatter(X[0, :], Y)
    plt.xlabel('△p(pixel)', fontsize=16)
    plt.ylabel('△x(step)', fontsize=16)
    plt.show()
    beta_leastsquares = beta_leastsquare(X, Y)
    for i in range(num_epoch):
        beta_huber = beta_updata(beta_huber, lr, X, Y, sigma1, 1)
        beta_tukey = beta_updata(beta_tukey, lr, X, Y, sigma2, 2)
    beta_huber = np.squeeze(beta_huber)
    beta_tukey = np.squeeze(beta_tukey)
    plt.figure(figsize=(6, 10))
    plt.scatter(X[0, :], Y)
    plot_fit(X, beta_huber, beta_leastsquares, beta_tukey)
    plt.xlabel('△p(pixel)', fontsize=16)
    plt.ylabel('△x(step)', fontsize=16)
    plt.legend(["Huber", "LS", "Tukey"], loc="upper right")
    plt.show()
    print("--------------------------参数--------------------------")
    print("beta_leastsquares:", beta_leastsquares)
    print("beta_huber:", beta_huber.squeeze())
    print("beta_tukey:", beta_tukey.squeeze())
    res1, res2, res3, Len, differ1, differ2, differ1_max, differ2_max = res(X, Y, beta_leastsquares, beta_huber, beta_tukey)
    print("outliers_notremove", "Leastsquares:", res1, "huber:", res2, "tukey:", res3)
    res1, res2, res3, Len, differ1, differ2, differ1_max, differ2_max = res(X, Y, beta_leastsquares, beta_huber, beta_tukey, outliers_remove=True)
    print("outliers_remove", "Leastsquares:", res1, "huber:", res2, "tukey:", res3)
    # plt.scatter(1, res1, c='b')
    # plt.scatter(2, res2, c='r')
    # plt.scatter(1, differ1_max, c='k')
    # plt.scatter(2, differ2_max, c='c')
    # plt.scatter(1, differ1/Len, c='g')
    # plt.scatter(2, differ2/Len, c='y')
    plt.bar(np.arange(3), [res1, differ1_max, differ1/Len], width=0.2, color='r', label='LS')
    plt.bar(np.arange(3)+0.2, [res2, differ2_max, differ2 / Len], width=0.2, color='b', label='Huber')
    plt.legend(loc="upper right", prop={"size": 12, })
    plt.xticks(np.arange(3)+0.1, ["MSE", "RES", "MAE"], size=16)
    plt.ylim(6, 150)
    # plt.text(1, res1, "{:.2f}".format(res1), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(2, res2, "{:.2f}".format(res2), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(1, differ1_max, "{:.2f}".format(differ1_max), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(2, differ2_max, "{:.2f}".format(differ2_max), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(1, differ1/Len, "{:.2f}".format(differ1/Len), fontsize=12, verticalalignment='bottom', horizontalalignment='left')
    # plt.text(2, differ2/Len, "{:.2f}".format(differ2/Len), fontsize=12, verticalalignment='bottom', horizontalalignment='left')

    plt.xlabel('Method', fontsize=16)
    # plt.xticks(np.arange(0, 4, 1))
    plt.ylabel('MSE/RES/MAE', fontsize=16)
    # plt.legend(['LS_MSE', "Huber_MSE", "LS_RES", "Huber_RES", "LS_MAE", "Huber_MAE"], loc='upper right')
    plt.show()
    print("--------------------------参数--------------------------")
    return beta_leastsquares, beta_huber, beta_tukey

def x_gap_method1(Data):
    huber_gap = []
    least_gap = []
    for i,data in enumerate(Data):
        x_center = data[0]
        y_noncenter = data[1]
        p = data[2]
        x_center_huber = para[0] * np.sin(para[1] * (y_noncenter - (\
                    beta_huber[0] * (p - 1000) + beta_huber[1])) + para[2]) + para[3]
        x_center_least = para[0] * np.sin(para[1] * (y_noncenter - (\
                    beta_leastsquares[0] * (p - 1000) + beta_leastsquares[1])) + para[2]) + para[3]
        huber_gap.append(x_center-x_center_huber)
        least_gap.append(x_center-x_center_least)
    print("Huber差距平均值:", np.mean(np.abs(huber_gap)))
    print("Least差距平均值:", np.mean(np.abs(least_gap)))
    return np.abs(huber_gap), np.abs(least_gap)

def x_gap_method2(Data):
    huber_gap = []
    least_gap = []
    for i,data in enumerate(Data):
        x_center = data[0]
        y_noncenter = data[1]
        p = data[2]
        x_center_huber = para[0] * np.sin(para[1] * y_noncenter + para[2]) + para[3] - (
                    beta_huber[0] * (p - 1000) + beta_huber[1])
        x_center_least = para[0] * np.sin(para[1] * y_noncenter + para[2]) + para[3] - (
                    beta_leastsquares[0] * (p - 1000) + beta_leastsquares[1])
        # print(x_center_least,x_center)
        huber_gap.append(x_center-x_center_huber)
        least_gap.append(x_center-x_center_least)
    print("Huber差距平均值:", np.mean(np.abs(huber_gap)))
    print("Least差距平均值:", np.mean(np.abs(least_gap)))
    return np.abs(huber_gap), np.abs(least_gap)

def showcurve3(y1, y2):
    x1 = np.arange(len(y1))
    x2 = np.arange(len(y2))
    plt.xlabel('index')
    plt.ylabel('gap')
    plt.plot(x1, y1, '-o', label='huber_gap', color='b')
    plt.plot(x2, y2, '-o', label='least_gap',color='g')
    plt.legend(loc='upper right')
    plt.show()

def showcurve4(y1, y2):
    x1 = np.arange(len(y1))
    x2 = np.arange(len(y2))
    plt.xlabel('index')
    plt.ylabel('gap')
    plt.plot(x1, y1, '-o', label='y_noncenter', color='b')
    plt.plot(x2, y2, '-o', label='x_center',color='g')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":

    lr = 0.001
    beta_huber = np.zeros((2, 1))-0.1
    beta_tukey = np.zeros((2, 1))-0.1
    sigma1 = 0.001
    sigma2 = 0.001
    num_epoch = 150
    machine = 4
    raster = "150_4000"
    if machine == 0:
        print("不执行machine")
    else:
        """1200光栅"""
        x_1200_1 = np.array([0, 253.65, 312.57, 365.02, 404.66, 576.96, 614.31, 640.22, 703.24, 743.89, 965.779])#样机1
        y_1200_1 = np.array([327, 14456, 17740, 20659, 22862, 32421, 34496, 35934, 39422, 41674, 53934])#样机1
        x_1200_2 = np.array([0, 253.6,312.57,313.16, 365.02,404.66,435.84,546.07,576.96,585.25, 614.31,640.22, 703.24,743.89,842.47,965.779])#样机2
        y_1200_2 = np.array([3708, 17836, 21096, 21130, 23998, 26191, 27916, 34014, 35725, 36178, 37772, 39194, 42640, 44862, 50242, 56953])#样机2
        xx = np.array([253.65,312.57,435.84,546.07,640.22,866.79])
        yy = np.array([113,113,112,111,108,97])
        x_1200_3 = np.array([0, 253.65,312.57, 365.02,404.66,435.84,576.96,594.48,614.31,640.22,703.24,743.89,965.779])#样机3
        y_1200_3 = np.array([4456, 18580, 21837, 24733, 26919, 28636, 36377, 37336, 38419, 39832, 43257, 45469, 57482])     #样机3
        x_1200_4 = np.array([253.65,312.57,365.02,404.66,435.84,546.07,576.96,585.25,604.22,703.24,750.39,840.82,842.47])#样机4
        y_1200_4 = np.array([16963,20300,23130,25324,27045,30991, 34834, 35289, 38318, 41766, 45059, 49272, 49362])#样机4



        print("x_1200: 波长   y_1200: 设置值")
        x = locals()['x_1200_{}'.format(machine)]
        y = locals()['y_1200_{}'.format(machine)]
        plt.scatter(x,y)
        plt.show()
        para_, target_func1_ = Fittingcurve1(x,y, '波长', '设置值')
        # y = np.apply_along_axis(Fittingcurve1_, 0, x, para_)
        para, target_func1 = Fittingcurve1(y,x, '波长', '设置值')
        showcurve1(y,x, para, target_func1, '2400', 'Feedback Values', 'Wavelengths')

        y = para_[0] * np.sin(para_[1] * 585.25 + para_[2]) + para_[3]
        print("y:",y)
        x = para[0] * np.sin(para[1] * y + para[2]) + para[3]
        print("x:",x)
        print("x-x_real:", x-585.25)
        delta_y_1 = np.array([-456,-256,-56,0,144,344,494
    ,-440
    ,-240
    ,-140
    ,0
    ,160
    ,360
    ,510
    ,-459
    ,-259
    ,-59
    ,0
    ,141
    ,341
    ,491
    ,-462
    ,-262
    ,-62
    ,0
    ,138
    ,338
    ,488
    ,-471
    ,-371
    ,-171
    ,0
    ,100
    ,250
    ,450
    ,-450
    ,-300
    ,-100
    ,0
    ,150
    ,300
    ,450
    ,-400
    ,-250
    ,-100
    ,0
    ,150
    ,300
    ,450
    ,-400
    ,-250
    ,-50
    ,0
    ,150
    ,300
    ,450
    ,-400
    ,-250
    ,-100
    ,0
    ,150
    ,300
    ,450
    ,-400
    ,-250
    ,-100
    ,0
    ,150
    ,300
    ,400
    ])  # 样机1
        delta_p_1 = np.array([862
    ,490
    ,120
    ,0
    ,-250
    ,-620
    ,-892
    ,824
    ,453
    ,264
    ,0
    ,-290
    ,-653
    ,-924
    ,859
    ,483
    ,107
    ,0
    ,-259
    ,-625
    ,-896
    ,865
    ,488
    ,115
    ,0
    ,-255
    ,-621
    ,-893
    ,909
    ,714
    ,329
    ,0
    ,-186
    ,-469
    ,-844
    ,871
    ,579
    ,191
    ,0
    ,-293
    ,-572
    ,-856
    ,774
    ,479
    ,189
    ,0
    ,-303
    ,-581
    ,-866
    ,791
    ,494
    ,103
    ,0
    ,-286
    ,-571
    ,-867
    ,801
    ,498
    ,199
    ,0
    ,-286
    ,-574
    ,-865
    ,866
    ,540
    ,219
    ,0
    ,-304
    ,-614
    ,-819
    ])  # 样机1
        delta_p_1 = np.vstack((delta_p_1, np.ones_like(delta_p_1)))
        delta_y_2 = np.array(
            [-436
    ,-236
    ,-86
    ,0
    ,64
    ,164
    ,364
    ,-496
    ,-296
    ,-96
    ,0
    ,34
    ,134
    ,334
    ,-330
    ,-130
    ,-34
    ,0
    ,100
    ,300
    ,400
    ,-398
    ,-198
    ,-98
    ,0
    ,102
    ,202
    ,402
    ,-491
    ,-291
    ,-91
    ,0
    ,109
    ,209
    ,409
    ,-466
    ,-266
    ,-66
    ,0
    ,84
    ,184
    ,384
    ,-464
    ,-264
    ,-64
    ,0
    ,86
    ,186
    ,386
    ,-425
    ,-225
    ,-125
    ,0
    ,75
    ,175
    ,375
    ,-78
    ,0
    ,122
    ,222
    ,422
    ,-372
    ,-172
    ,-72
    ,0
    ,128
    ,228
    ,428
    ,-394
    ,-194
    ,-94
    ,0
    ,106
    ,206
    ,406
    ,-240
    ,-140
    ,-40
    ,0
    ,160
    ,260
    ,360
    ,-462
    ,-262
    ,-62
    ,0
    ,88
    ,238
    ,438
    ,-342
    ,-142
    ,-42
    ,0
    ,158
    ,358
    ,-353
    ,-153
    ,-53
    ,0
    ,97
    ,347
    ])  # 样机2
        delta_p_2 = np.array([818
    ,442
    ,162
    ,0
    ,-115
    ,-302
    ,-673
    ,936
    ,556
    ,180
    ,0
    ,-64
    ,-250
    ,-618
    ,619
    ,224
    ,62
    ,0
    ,-188
    ,-557
    ,-740
    ,754
    ,374
    ,185
    ,0
    ,-188
    ,-373
    ,-741
    ,935
    ,552
    ,172
    ,0
    ,-207
    ,-395
    ,-765
    ,893
    ,508
    ,126
    ,0
    ,-160
    ,-350
    ,-728
    ,900
    ,508
    ,122
    ,0
    ,-166
    ,-359
    ,-742
    ,833
    ,442
    ,246
    ,0
    ,-147
    ,-343
    ,-716
    ,149
    ,0
    ,-240
    ,-437
    ,-823
    ,738
    ,340
    ,143
    ,0
    ,-252
    ,-448
    ,-837
    ,786
    ,384
    ,187
    ,0
    ,-209
    ,-406
    ,-799
    ,485
    ,282
    ,82
    ,0
    ,-321
    ,-522
    ,-720
    ,942
    ,530
    ,124
    ,0
    ,-181
    ,-485
    ,-883
    ,715
    ,295
    ,88
    ,0
    ,-324
    ,-730
    ,776
    ,334
    ,116
    ,0
    ,-207
    ,-738
    ])  # 样机2
        delta_p_2 = np.vstack((delta_p_2, np.ones_like(delta_p_2)))
        delta_y_3 = np.array([-480
    ,-280
    ,-80
    ,0
    ,200
    ,350
    ,420
    ,-337
    ,-237
    ,-137
    ,0
    ,163
    ,363
    ,463
    ,543
    ,-353
    ,-233
    ,-133
    ,-43
    ,0
    ,167
    ,267
    ,467
    ,-439
    ,-319
    ,-119
    ,0
    ,131
    ,321
    ,481
    ,-436
    ,-236
    ,-86
    ,0
    ,202
    ,344
    ,484
    ,-463
    ,-263
    ,-83
    ,0
    ,200
    ,337
    ,487
    ,-377
    ,-177
    ,-77
    ,0
    ,123
    ,323
    ,473
    ,520
    ,-436
    ,-336
    ,-136
    ,0
    ,164
    ,364
    ,479
    ,-419
    ,-219
    ,-69
    ,0
    ,181
    ,381
    ,481
    ,-532
    ,-332
    ,-132
    ,0
    ,14
    ,168
    ,458
    ,-457
    ,-257
    ,-107
    ,0
    ,200
    ,343
    ,493
    ,-469
    ,-269
    ,-69
    ,0
    ,201
    ,331
    ,481
    ,-416
    ,-216
    ,-76
    ,0
    ,204
    ,354
    ,484
    ,-432
    ,-232
    ,-82
    ,0
    ,118
    ,318
    ,458
    ]) #样机3
        delta_p_3 = np.array([899
    ,523
    ,151
    ,0
    ,-365
    ,-644
    ,-770
    ,632
    ,443
    ,256
    ,0
    ,-300
    ,-668
    ,-852
    ,-996
    ,663
    ,438
    ,251
    ,82
    ,0
    ,-308
    ,-494
    ,-861
    ,834
    ,605
    ,227
    ,0
    ,-243
    ,-601
    ,-899
    ,831
    ,448
    ,164
    ,0
    ,-376
    ,-641
    ,-899
    ,885
    ,505
    ,160
    ,0
    ,-379
    ,-635
    ,-915
    ,732
    ,342
    ,148
    ,0
    ,-237
    ,-615
    ,-898
    ,-987
    ,854
    ,658
    ,266
    ,0
    ,-314
    ,-695
    ,-913
    ,825
    ,429
    ,135
    ,0
    ,-335
    ,-728
    ,-917
    ,956
    ,658
    ,260
    ,0
    ,-28
    ,-324
    ,-876
    ,899
    ,503
    ,208
    ,0
    ,-387
    ,-663
    ,-950
    ,954
    ,545
    ,140
    ,0
    ,-397
    ,-651
    ,-944
    ,861
    ,446
    ,157
    ,0
    ,-410
    ,-710
    ,-966
    ,956
    ,507
    ,179
    ,0
    ,-253
    ,-677
    ,-963
    ]) #样机3
        delta_p_3 = np.vstack((delta_p_3, np.ones_like(delta_p_3)))
        delta_x_3 = np.array([-9.23,-6.29,0,3.07,8.52,-6.88,-4.69,0,2.05,7.4,-6.79,-2.96,0,4.86,6.78,-2.64,0,6.61,0,3.87])
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
        delta_p_4 = np.vstack((delta_p_4, np.ones_like(delta_p_4)))
        p = locals()['delta_p_{}'.format(machine)]
        y = locals()['delta_y_{}'.format(machine)]
        beta_leastsquares, beta_huber_1200, beta_tukey_1200 = main_ht(p, y, lr, num_epoch, sigma1,sigma2,
                                                                          beta_huber, beta_tukey)


        Data_1 = np.array([[253.65,253.65,253.65,253.65,253.65,253.65,313.16,313.16,313.16,313.16,313.16,313.16,404.66,
                            404.66,404.66,404.66,404.66,435.84,435.84,435.84,435.84,435.84,546.07,546.07,546.07,546.07,
                            546.07,579.07,579.07,579.07,579.07,579.07,614.31,614.31,630.48,630.48,667.83,667.83,724.52,
                            724.52],[17000,17200,17400,17510,17700,17900,20473,20600,20763,20797,21000,21200,25400,25600,
                            25842,26050,26250,27250,27500,27555,27750,27950,33197,33400,33605,33800,34000,34900,35100,35297,
                            35410,35600,37150,37337,38200,38219,39957,40256,43200,43349],[1985,1645,1209,1000,650,279,1603,
                            1367,1062,1000,622,252,1837,1458,1000,610,236,1574,1104,1000,633,258,1786,1393,1000,628,248,1992,
                            1602,1219,1000,636,1362,1000,1035,1000,1590,1000,1296,1000]]).T    #样机1
        Data_4 = np.array([[253.65,253.65,253.65,253.65,253.65,253.65,313.16,313.16,313.16,313.16,313.16,313.16,404.66,
                            404.66,404.66,404.66,404.66,435.84,435.84,435.84,435.84,435.84,546.07,546.07,546.07,546.07,
                            546.07,579.07,579.07,579.07,579.07,579.07,614.31,614.31,630.48,630.48,667.83,667.83,724.52,
                            724.52],[17000,17200,17400,17510,17700,17900,20473,20600,20763,20797,21000,21200,25400,25600,
                            25842,26050,26250,27250,27500,27555,27750,27950,33197,33400,33605,33800,34000,34900,35100,35297,
                            35410,35600,37150,37337,38200,38219,39957,40256,43200,43349],[1985,1645,1209,1000,650,279,1603,
                            1367,1062,1000,622,252,1837,1458,1000,610,236,1574,1104,1000,633,258,1786,1393,1000,628,248,1992,
                            1602,1219,1000,636,1362,1000,1035,1000,1590,1000,1296,1000]]).T    #样机4
        print(Data_1.shape)
        # Data_3_y_compute = []
        # for i in [585.25,588.19,594.48,597.55,603,607.43,609.62,614.31,616.36,621.71,626.65,630.48,633.44,638.3,640.22,650.65,653.29,659.9,667.83,671.7]:
        #     Data_3_y_compute.append(para_[0] * np.sin(para_[1] * i + para_[2]) + para_[3])
        # print(Data_3_y_compute)
        # Data_3 = np.array([[585.25,588.19,594.48,597.55,603,607.43,609.62,614.31,616.36,621.71,626.65,630.48,633.44,638.3,640.22,650.65,653.29,659.9,667.83,671.7],
        #                   [39,344,1000,1325,1905,281,508,1000,1217,1789,287,688,1000,1517,1724,722,1000,1708,1000,1416]]).T                                        # 样机3
        # Data_3 = np.insert(Data_3, 1, values=Data_3_y_compute, axis=1)
        # print(Data_3)
        huber_gap, least_gap = x_gap_method1(Data_4)
        showcurve3(huber_gap, least_gap)
        showcurve4(Data_4[:,1], Data_4[:,0]*50)

        #已知非中心求中心
        # p_center = "中心像元值"
        # p_noncenter = "非中心像元值  已知"
        # x_noncenter = "非中心波长值  已知"
        # y_center_huber = para_[0] * np.sin(para_[1] * x_noncenter + para_[2]) + para_[3] - (beta_huber[0] * (p_center - p_noncenter) + beta_huber[1])
        # x_center_huber = para[0] * np.sin(para[1] * y_center_huber + para[2]) + para[3]



        """2400光栅"""
        x_2400_1 = np.array([0, 253.65, 312.57, 365.02, 404.66, 435.84, 546.07, 576.96])  # 样机1
        y_2400_1 = np.array([388, 28619, 35156, 40968, 45358, 48804, 60960, 64347])        # 样机1
        x_2400_2 = np.array([0, 253.65, 312.57, 313.16, 365.02, 404.66, 435.84, 546.07, 576.96, 579.07]) #样机2
        y_2400_2 = np.array([3681, 31859, 38353, 38421, 44095, 48428, 51826, 63807, 67142, 67372])         #样机2
        x_2400_3 = np.array([0, 253.65, 312.57, 365.02,404.66,435.84, 546.07, 579.07])  #样机3
        y_2400_3 = np.array([4449, 32564, 39015, 44723, 49031, 52409, 64278, 67822])  #样机3
        x_2400_4 = np.array([253.65, 312.57, 313.16, 365.02, 404.66, 435.84, 546.07])  # 样机4
        y_2400_4 = np.array([30862, 37361, 37428, 43108, 47440, 50839, 62791])        # 样机4
        print("x_2400: 波长   y_2400: 设置值")
        x = locals()['x_2400_{}'.format(machine)]
        y = locals()['y_2400_{}'.format(machine)]
        para_2400_, target_func1_2400_ = Fittingcurve1(x, y, '波长', '设置值')
        # y = np.apply_along_axis(Fittingcurve1_, 0, x, para_)
        para_2400, target_func1_2400 = Fittingcurve1(y, x, '波长', '设置值')

        showcurve1(y, x, para_2400, target_func1_2400, '2400', 'Feedback Values(step)', 'Wavelengths(nm)')

        delta_y_2400_1 = np.array(
            [-499
    ,-319
    ,-119
    ,0
    ,231
    ,381
    ,507
    ,-456
    ,-256
    ,-56
    ,0
    ,244
    ,444
    ,544
    ,-468
    ,-268
    ,-68
    ,0
    ,202
    ,332
    ,492
    ,-471
    ,-258
    ,-58
    ,0
    ,242
    ,442
    ,488
    ,-454
    ,-254
    ,-54
    ,0
    ,146
    ,346
    ,446
    ,-428
    ,-256
    ,-109
    ,0
    ,190
    ,320
    ,444
    ,-379
    ,-236
    ,-52
    ,0
    ,156
    ,344
    ,388
    ])  # 样机1
        delta_p_2400_1 = np.array(
            [396
    ,601
    ,225
    ,0
    ,-430
    ,-709
    ,-945
    ,872
    ,488
    ,105
    ,0
    ,-465
    ,-839
    ,-901
    ,916
    ,525
    ,133
    ,0
    ,-392
    ,-640
    ,-947
    ,951
    ,519
    ,118
    ,0
    ,-476
    ,-867
    ,-958
    ,935
    ,521
    ,113
    ,0
    ,-293
    ,-698
    ,-896
    ,972
    ,577
    ,242
    ,0
    ,-326
    ,-710
    ,-980
    ,885
    ,551
    ,119
    ,0
    ,-361
    ,-788
    ,-885
    ])  # 样机1
        delta_p_2400_1 = np.vstack((delta_p_2400_1, np.ones_like(delta_p_2400_1)))

        delta_y_2400_2 = np.array(
            [-359
    ,-259
    ,-109
    ,0
    ,141
    ,241
    ,341
    ,-353
    ,-153
    ,-53
    ,0
    ,68
    ,247
    ,347
    ,447
    ,-421
    ,-221
    ,-121
    ,-68
    ,0
    ,179
    ,279
    ,379
    ,-395
    ,-195
    ,-95
    ,0
    ,105
    ,305
    ,405
    ,-428
    ,-228
    ,-128
    ,0
    ,72
    ,172
    ,372
    ,-426
    ,-226
    ,-76
    ,0
    ,74
    ,174
    ,374
    ,-407
    ,-207
    ,-57
    ,0
    ,93
    ,293
    ,393
    ,-292
    ,-92
    ,0
    ,158
    ,230
    ,358
    ,-322
    ,-222
    ,-72
    ,0
    ,128
    ,328
    ])  # 样机2
        delta_p_2400_2 = np.array(
            [687
    ,493
    ,207
    ,0
    ,-268
    ,-456
    ,-642
    ,693
    ,299
    ,105
    ,0
    ,-132
    ,-478
    ,-670
    ,-860
    ,821
    ,430
    ,234
    ,129
    ,0
    ,-350
    ,-541
    ,-732
    ,799
    ,391
    ,191
    ,0
    ,-210
    ,-606
    ,-802
    ,880
    ,469
    ,260
    ,0
    ,-147
    ,-349
    ,-749
    ,895
    ,474
    ,159
    ,0
    ,-154
    ,-360
    ,-767
    ,951
    ,480
    ,133
    ,0
    ,-213
    ,-667
    ,-891
    ,716
    ,225
    ,0
    ,-376
    ,-548
    ,-845
    ,785
    ,537
    ,173
    ,0
    ,-310
    ,-786

    ])  # 样机2
        delta_p_2400_2 = np.vstack((delta_p_2400_2, np.ones_like(delta_p_2400_2)))

        delta_y_2400_3 = np.array(
            [-464
    ,-264
    ,-114
    ,0
    ,136
    ,286
    ,436
    ,-296
    ,-165
    ,-65
    ,0
    ,185
    ,385
    ,485
    ,-423
    ,-223
    ,-23
    ,0
    ,177
    ,377
    ,477
    ,-331
    ,-181
    ,-81
    ,0
    ,201
    ,369
    ,469
    ,-409
    ,-209
    ,-9
    ,0
    ,211
    ,411
    ,461
    ,-373
    ,-128
    ,-48
    ,0
    ,202
    ,342
    ,422
    ,-324
    ,-122
    ,-22
    ,0
    ,128
    ,278
    ,378
    ])  # 样机3
        delta_p_2400_3 = np.array(
            [870
    ,498
    ,215
    ,0
    ,-255
    ,-535
    ,-810
    ,573
    ,321
    ,128
    ,0
    ,-355
    ,-735
    ,-922
    ,848
    ,447
    ,47
    ,0
    ,-347
    ,-737
    ,-930
    ,673
    ,367
    ,163
    ,0
    ,-406
    ,-737
    ,-937
    ,858
    ,435
    ,23
    ,0
    ,-430
    ,-835
    ,-935
    ,872
    ,294
    ,111
    ,0
    ,-460
    ,-776
    ,-953
    ,778
    ,297
    ,56
    ,0
    ,-297
    ,-648
    ,-878
    ])  # 样机3
        delta_p_2400_3 = np.vstack((delta_p_2400_3, np.ones_like(delta_p_2400_3)))

        delta_y_2400_4 = np.array(
            [-362
    ,-262
    ,-62
    ,0
    ,138
    ,338
    ,388

    ,-463
    ,-327
    ,-159
    ,0
    ,67
    ,232
    ,422
    ,471

    ,-394
    ,-226
    ,-67
    ,0
    ,165
    ,355
    ,404

    ,-470
    ,-320
    ,-156
    ,0
    ,192
    ,392
    ,492

    ,-472
    ,-305
    ,-134
    ,0
    ,210
    ,321
    ,466

    ,-455
    ,-343
    ,-120
    ,0
    ,161
    ,261
    ,361

    ,-391
    ,-191
    ,-91
    ,0
    ,9
    ,109
    ,323
    ,408
    ])  # 样机4
        delta_p_2400_4 = np.array(
            [785
    ,500
    ,119
    ,0
    ,-260
    ,-640
    ,-733

    ,912
    ,642
    ,313
    ,0
    ,-130
    ,-451
    ,-820
    ,-914

    ,776
    ,446
    ,131
    ,0
    ,-322
    ,-691
    ,-785

    ,951
    ,646
    ,314
    ,0
    ,-377
    ,-770
    ,-964

    ,979
    ,629
    ,275
    ,0
    ,-425
    ,-651
    ,-939

    ,961
    ,723
    ,252
    ,0
    ,-330
    ,-535
    ,-738
    ,918
    ,451
    ,211
    ,0
    ,-14
    ,-245
    ,-726
    ,-916
    ])  # 样机4
        delta_p_2400_4 = np.vstack((delta_p_2400_4, np.ones_like(delta_p_2400_4)))

        p = locals()['delta_p_2400_{}'.format(machine)]
        y = locals()['delta_y_2400_{}'.format(machine)]
        beta_leastsquares_2400, beta_huber_2400, beta_tukey_2400 = main_ht(p, y, lr, num_epoch, sigma1, sigma2, beta_huber, beta_tukey)
        #已知非中心求中心
        # p_2400_center = "中心像元值"
        # p_2400_noncenter = "非中心像元值"
        # x_2400_noncenter = "非中心波长值"
        # y_center_leastsquares_2400 = para_2400_[0] * np.sin(para_2400_[1] * x_2400_noncenter + para_2400_[2]) + para_2400_[3] - (beta_leastsquares_2400[0] * (p_2400_center - p_2400_noncenter) + beta_leastsquares_2400[1])
        # x_center_leastsquares_2400 = para_2400[0] * np.sin(para_2400[1] * y_center_leastsquares_2400 + para_2400[2]) + para_2400[3]

    if raster == 0:
        print("不执行raster")
    elif raster == "75":
        #红外光谱的 75

        delta_p_75_ir = np.array([187,127,70,57,28,0,-27,-56,-89,-116,-176,-236,95,37,0,-35,-91,-109,113,99,37,0,-37,-96,259,124,0,-120,-235])
        delta_x_75_ir = np.array([160,110,60,50,25,0,-25,-50,-75,-100,-150,-200,80,30,0,-30,-80,-92,92,80,30,0,-30,-80,200,100,0,-100,-200])
        polypara_75_ir = Fittingcurve2(delta_p_75_ir, delta_x_75_ir, 'delta_p_75_ir', 'delta_x_75_ir')
        showcurve2(delta_p_75_ir, delta_x_75_ir, polypara_75_ir, '75_ir', 'delta_p_75_ir', 'delta_x_75_ir')
        p_noncenter_ir =8635 + (polypara_75_ir[0] * (370 - (342+347+350+348)/4) + polypara_75_ir[1])
        print(p_noncenter_ir,'p_noncenter_ir')
    elif raster == "150_2000":
        #红外光谱的 150/2000
        print("----------------------------------------红外光谱的 150/2000----------------------------------------")
        delta_p_150_2000_ir = np.array([0
,48
,141
,193
,-213
,-147
,-64
,0
])
        delta_x_150_2000_ir = np.array([0
,24
,63
,86
,-86
,-62
,-23
,0
])

        polypara_150_2000_ir = Fittingcurve2(delta_p_150_2000_ir, delta_x_150_2000_ir, 'delta_p_150_2000_ir', 'delta_x_150_2000_ir')
        showcurve2(delta_p_150_2000_ir, delta_x_150_2000_ir, polypara_150_2000_ir, '150_2000_ir', 'delta_p_150_2000_ir', 'delta_x_150_2000_ir')

        p = locals()['delta_p_{}_ir'.format(raster)]
        x = locals()['delta_x_{}_ir'.format(raster)]
        para, target_func1 = Fittingcurve1(p, x, '像元差', '波长差')
        showcurve1(p, x, para, target_func1, '红外光谱的 150/2000', '像元差', '波长差')

        delta_p_150_2000_ir = np.vstack((delta_p_150_2000_ir, np.ones_like(delta_p_150_2000_ir)))
        p = locals()['delta_p_{}_ir'.format(raster)]
        beta_leastsquares_ir, beta_huber_ir, beta_tukey_ir0 = main_ht(p, x, lr, num_epoch, sigma, beta_huber, beta_tukey)

        x = np.array([3244, 3509])
        p = np.array([362, 316])
        polypara_150_2000_ir = Fittingcurve2(x, p, 'x', 'p')
        showcurve2(x, p, polypara_150_2000_ir, '150_2000_ir', 'x', 'p')
        print("----------------------------------------红外光谱的 150/2000----------------------------------------")

    else:

        # 红外光谱的 150/4000
        print("----------------------------------------红外光谱的 150/4000----------------------------------------")
        delta_p_150_4000_ir = np.array([0
,61
,146
,201
,-199
,-166
,-58
,0
])
        delta_x_150_4000_ir = np.array([0
,24
,63
,86
,-86
,-62
,-23
,0
])

        polypara_150_4000_ir = Fittingcurve2(delta_p_150_4000_ir, delta_x_150_4000_ir, 'delta_p_150_4000_ir',
                                             'delta_x_150_4000_ir')
        showcurve2(delta_p_150_4000_ir, delta_x_150_4000_ir, polypara_150_4000_ir, '150_4000_ir', 'delta_p_150_4000_ir',
                   'delta_x_150_4000_ir')

        p = locals()['delta_p_{}_ir'.format(raster)]
        x = locals()['delta_x_{}_ir'.format(raster)]
        para, target_func1 = Fittingcurve1(p, x, '像元差', '波长差')
        showcurve1(p, x, para, target_func1, '红外光谱的 150/4000', '像元差', '波长差')

        delta_p_150_4000_ir = np.vstack((delta_p_150_4000_ir, np.ones_like(delta_p_150_4000_ir)))
        p = locals()['delta_p_{}_ir'.format(raster)]
        beta_leastsquares_ir, beta_huber_ir, beta_tukey_ir0 = main_ht(p, x, lr, num_epoch, sigma, beta_huber,
                                                                      beta_tukey)

        x = np.array([3244, 3509])
        p = np.array([368, 317])
        polypara_150_4000_ir = Fittingcurve2(x, p, 'x', 'p')
        showcurve2(x, p, polypara_150_4000_ir, '150_4000_ir', 'x', 'p')
        print("----------------------------------------红外光谱的 150/4000----------------------------------------")