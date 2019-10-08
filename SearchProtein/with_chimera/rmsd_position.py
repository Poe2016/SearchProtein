# -*- coding:utf-8 -*-
import numpy as np
from scipy import spatial
import argparse

parser = argparse.ArgumentParser(description="position", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-f1", "--file1", type=str, help="PDB filename1")
parser.add_argument("-f2", "--file2", type=str, help="PDB filename2")


class Molecule(object):

    def __init__(self, filename, data_path):
        '''
        分子类初始化，通过分子文件来初始化
        :param filename:
        '''
        self.filename = filename
        self.data_path = data_path
        self.natoms = 0
        with open(filename) as f:
            for line in f:
                if line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
                    self.natoms += 1
        print('aaaa   ', self.natoms)
        self.coords = np.zeros((self.natoms, 3))
        self.nelectrons = np.zeros((self.natoms), dtype=int)
        print(filename)
        with open(filename) as f:
            atom = 0
            for line in f:
                if line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
                    self.coords[atom, 0] = float(line[30:38])
                    self.coords[atom, 1] = float(line[38:46])
                    self.coords[atom, 2] = float(line[46:54])
                    atom += 1


def position_state(data1, data2):
    '''
    计算两个分子是否重叠，或者相隔很远
    :param pdb:
    :param cx0:
    :param cy0:
    :param cz0:
    :param cx1:
    :param cy1:
    :param cz1:
    :return:
    '''
    state = ['overlap', 'isolate', 'normal']
    dist_min_threshold = 1
    dist_max_threshold = 4
    # 计算两个分子中每个原子的距离
    dist_map = spatial.distance.cdist(data1, data2)
    min_distance = np.min(dist_map)
    num_ovlerlap = len(dist_map[dist_map<dist_min_threshold])
    if min_distance < dist_min_threshold:
        print('overlap')
    elif min_distance > dist_max_threshold:
        print('isolate')
    else:
        print('normal')
    print(min_distance)
    print(num_ovlerlap)


args = parser.parse_args()

if __name__ == "__main__":
    data1 = np.loadtxt(args.file1)[:, :4]
    data2 = np.loadtxt(args.file2)[:, :4]
    # data1 = np.loadtxt('/Users/wyf/Desktop/sbw/version_manager/auto_fit_902/test/exp/data1.dat')[:, :4]
    # data2 = np.loadtxt('/Users/wyf/Desktop/sbw/version_manager/auto_fit_902/test/exp/data2.dat')[:, :4]
    position_state(data1, data2)
