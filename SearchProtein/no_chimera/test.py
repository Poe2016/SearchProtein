from handle import PDB
import numpy as np
from intensity import cal_intensity as ic
import matplotlib.pyplot as plt
from multiprocessing import Pool


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


if __name__ == "__main__":
    # profile_path = 'test/iq0.dat'
    # iq_path = 'test/out_iq_0.dat'
    # data_display(profile_path,iq_path)


    pdb=PDB('test/pdb/t1.pdb')
    # # hc(pdb, 'hc.dat')
    ic(pdb, 'ic2.dat')



    print('end')
