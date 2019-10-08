# -*- coding:utf-8 -*-
import numpy as np
# from intensity import cal_intensity as ic
import matplotlib.pyplot as plt
# from multiprocessing import Pool
from scipy import ndimage

def data_display(profile_path, iq_path):
    '''
    显示拟合曲线
    :param profile_path: 目标IQ文件路径
    :param iq_path: 预测IQ文件路径
    :return: void
    '''
    target_data = np.loadtxt(profile_path)[:, :2]
    iq_data = np.loadtxt(iq_path)[:, :2]
    plt.plot(target_data[:, 0],target_data[:, 1], label='target')
    plt.plot(iq_data[:, 0], iq_data[:, 1])
    plt.show()


def paral_cal():
    for i in range(100):
        print('s')

from scipy.interpolate import interp1d
from scipy.optimize import fmin


def score_curve(score_path, rmsd_path):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    data = np.loadtxt(score_path)
    rmsd_data = np.loadtxt(rmsd_path)
    count = 0
    old = 0
    # delete_list=[]
    # for i in range(len(data)):
    #     if data[i] != old:
    #         count = 1
    #         old = data[i]
    #     else:
    #         if count == 5:
    #             delete_list.append(i)
    #         else:
    #             count += 1
    # data = np.delete(data, delete_list)
    data = data[::10]
    x = np.arange(len(data))
    print(len(data))
    xfit = np.linspace(0, len(data), len(data)*10)
    # print(xfit)
    # create the interpolating function
    f = interp1d(x, data, kind='cubic', bounds_error=False)
    # plt.plot(xfit, f(xfit))
    lns1 = ax1.plot(xfit, f(xfit), label='diference')
    ax1.set_ylabel('difference')
    # ax1.set_title("Double Y axis")
    # ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    rmsd_x = np.arange(len(rmsd_data))
    # print(rmsd_x)
    rmsd_fit = np.linspace(0, len(data), len(data)*10)
    f2 = interp1d(rmsd_x, rmsd_data, kind='cubic', bounds_error=False)
    # plt.plot(rmsd_fit, f2(rmsd_fit))
    lns2 = ax2.plot(rmsd_fit, f2(rmsd_fit), 'r',label='rmsd')
    # ax2.set_xlim([0, np.e])
    ax2.set_ylabel('rmsd(Angstrom)')
    ax2.set_xlabel('Same X for both exp(-x) and ln(x)')
    # ax2.legend(loc='upper right')

    # 合并图例
    # added these three lines
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=0)

    plt.xlabel('time')
    # plt.ylabel('difference-rmsd')
    plt.savefig('diff_rmsd831.png')
    plt.show()







if __name__ == "__main__":
    score_curve('test829/iq_output_difference_list_out_iq_0.dat', 'test829/rmsd_rcd.dat')

    # profile_path = 'test829/t_iq_chi/target_iq0.dat'
    # iq_path = 'test829/iq_output/out_iq_0.dat'
    # data_display(profile_path, iq_path)


    # pdb=PDB('test829/pdb/t0.pdb')
    # # # # hc(pdb, 'hc.dat')
    # ic(pdb, 'ic2.dat')
    # print('ok')



    print('end')