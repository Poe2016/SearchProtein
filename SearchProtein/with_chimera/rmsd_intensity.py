# -*- coding:utf-8 -*-
import numpy as np
import sys
from scipy import spatial, ndimage
import argparse

parser = argparse.ArgumentParser(description="test", formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-f1", "--file1", type=str, help="data1 filename")
parser.add_argument("-f2", "--file2", type=str, help="data2 filename")
parser.add_argument("-t", "--target", type=str, help="target filename")
parser.add_argument("-o", "--output", type=str, help="output intensity filename")


electrons = {'H': 1, 'HE': 2, 'He': 2, 'LI': 3, 'Li': 3, 'BE': 4, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9,
             'NE': 10, 'Ne': 10, 'NA': 11, 'Na': 11, 'MG': 12, 'Mg': 12, 'AL': 13, 'Al': 13, 'SI': 14, 'Si': 14,
             'P': 15, 'S': 16, 'CL': 17, 'Cl': 17, 'AR': 18, 'Ar': 18, 'K': 19, 'CA': 20, 'Ca': 20, 'SC': 21, 'Sc': 21,
             'TI': 22, 'Ti': 22, 'V': 23, 'CR': 24, 'Cr': 24, 'MN': 25, 'Mn': 25, 'FE': 26, 'Fe': 26, 'CO': 27,
             'Co': 27, 'NI': 28, 'Ni': 28, 'CU': 29, 'Cu': 29, 'ZN': 30, 'Zn': 30, 'GA': 31, 'Ga': 31, 'GE': 32,
             'Ge': 32, 'AS': 33, 'As': 33, 'SE': 34, 'Se': 34, 'Se': 34, 'Se': 34, 'BR': 35, 'Br': 35, 'KR': 36,
             'Kr': 36, 'RB': 37, 'Rb': 37, 'SR': 38, 'Sr': 38, 'Y': 39, 'ZR': 40, 'Zr': 40, 'NB': 41, 'Nb': 41,
             'MO': 42, 'Mo': 42, 'TC': 43, 'Tc': 43, 'RU': 44, 'Ru': 44, 'RH': 45, 'Rh': 45, 'PD': 46, 'Pd': 46,
             'AG': 47, 'Ag': 47, 'CD': 48, 'Cd': 48, 'IN': 49, 'In': 49, 'SN': 50, 'Sn': 50, 'SB': 51, 'Sb': 51,
             'TE': 52, 'Te': 52, 'I': 53, 'XE': 54, 'Xe': 54, 'CS': 55, 'Cs': 55, 'BA': 56, 'Ba': 56, 'LA': 57,
             'La': 57, 'CE': 58, 'Ce': 58, 'PR': 59, 'Pr': 59, 'ND': 60, 'Nd': 60, 'PM': 61, 'Pm': 61, 'SM': 62,
             'Sm': 62, 'EU': 63, 'Eu': 63, 'GD': 64, 'Gd': 64, 'TB': 65, 'Tb': 65, 'DY': 66, 'Dy': 66, 'HO': 67,
             'Ho': 67, 'ER': 68, 'Er': 68, 'TM': 69, 'Tm': 69, 'YB': 70, 'Yb': 70, 'LU': 71, 'Lu': 71, 'HF': 72,
             'Hf': 72, 'TA': 73, 'Ta': 73, 'W': 74, 'RE': 75, 'Re': 75, 'OS': 76, 'Os': 76, 'IR': 77, 'Ir': 77,
             'PT': 78, 'Pt': 78, 'AU': 79, 'Au': 79, 'HG': 80, 'Hg': 80, 'TL': 81, 'Tl': 81, 'PB': 82, 'Pb': 82,
             'BI': 83, 'Bi': 83, 'PO': 84, 'Po': 84, 'AT': 85, 'At': 85, 'RN': 86, 'Rn': 86, 'FR': 87, 'Fr': 87,
             'RA': 88, 'Ra': 88, 'AC': 89, 'Ac': 89, 'TH': 90, 'Th': 90, 'PA': 91, 'Pa': 91, 'U': 92, 'NP': 93,
             'Np': 93, 'PU': 94, 'Pu': 94, 'AM': 95, 'Am': 95, 'CM': 96, 'Cm': 96, 'BK': 97, 'Bk': 97, 'CF': 98,
             'Cf': 98, 'ES': 99, 'Es': 99, 'FM': 100, 'Fm': 100, 'MD': 101, 'Md': 101, 'NO': 102, 'No': 102, 'LR': 103,
             'Lr': 103, 'RF': 104, 'Rf': 104, 'DB': 105, 'Db': 105, 'SG': 106, 'Sg': 106, 'BH': 107, 'Bh': 107,
             'HS': 108, 'Hs': 108, 'MT': 109, 'Mt': 109}


class Molecule(object):
    '''
    分子类
    '''

    def __init__(self, filename):
        '''
        分子类初始化，通过分子文件来初始化
        :param filename:
        '''
        self.filename = filename
        self.natoms = 0
        with open(filename) as f:
            for line in f:
                if line[0:4] == 'ATOM' or line[0:6] == 'HETATM':
                    self.natoms += 1
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
                    self.nelectrons[atom] = electrons[line[76:78].strip()]
                    atom += 1
        self.data = np.hstack((self.coords, self.nelectrons.reshape(-1, 1)))


def cal_intensity_temp(pdb):
    side = 48.
    nsamples = 10.
    voxel = side / nsamples
    halfside = side / 2
    n = int(side / voxel)
    if n % 2 == 1: n += 1
    dx = side / n
    x_ = np.linspace(-halfside, halfside, n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    rho = gauss(pdb, xyz=xyz, sigma=10)
    return rho, side


def gauss(pdb, xyz, sigma, eps=1e-6):
    '''

    :param pdb:
    :param xyz:
    :param sigma:
    :param eps:
    :return:
    '''
    coords = pdb[:, :3]
    elec = pdb[:, 3:4]
    n = int(round(xyz.shape[0] ** (1 / 3.)))
    sigma /= 4.
    dx = xyz[1, 2] - xyz[0, 2]
    shift = np.ones(3) * dx / 2.
    values = np.zeros((xyz.shape[0]))
    print(coords.shape[0])
    for i in range(coords.shape[0]):
        dist = spatial.distance.cdist(coords[None, i] - shift, xyz)
        dist *= dist
        values += elec[i] * 1. / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-dist[0] / (2 * sigma ** 2))
    return values.reshape(n, n, n)


def cal_intensity(pdb, out_iq_path='out_iq_path.dat'):
    '''

    :param pdb:
    :param out_iq_path:
    :return:
    '''
    ns = 1
    threshold = 0.0
    rho, side = cal_intensity_temp(pdb)
    if rho.shape[0] % 2 == 1:
        rho = rho[:-1, :-1, :-1]
    rho = np.copy(rho[::ns, ::ns, ::ns])
    rho[rho <= threshold] = 0
    halfside = side / 2
    nx, ny, nz = rho.shape[0], rho.shape[1], rho.shape[2]
    n = nx
    voxel = side / n
    n_orig = n
    dx = side / n
    dV = dx ** 3
    V = side ** 3
    x_ = np.linspace(-halfside, halfside, n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    df = 1. / side
    qx_ = np.fft.fftfreq(x_.size) * n * df * 2 * np.pi
    qz_ = np.fft.rfftfreq(x_.size) * n * df * 2 * np.pi
    qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins * 10)
    qbinsc = np.copy(qbins)
    qbinsc[1:] += qstep / 2.
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    qbin_args = np.copy(qbinsc)
    F = np.fft.fftn(rho)
    I3D = np.abs(F) ** 2
    Imean = ndimage.mean(I3D, labels=qbin_labels, index=np.unique(qbin_labels))
    qmax_to_use = np.max(qx_)
    qbinsc_to_use = qbinsc[qbinsc < qmax_to_use]
    isize = np.shape(Imean)
    qbinsc = qbinsc[:isize[0]]
    Imean_to_use = Imean[qbinsc < qmax_to_use]
    qbinsc = np.copy(qbinsc_to_use)
    Imean = np.copy(Imean_to_use)
    Iq = np.vstack((qbinsc, Imean, Imean * .03)).T
    np.savetxt(out_iq_path, Iq, delimiter=' ', fmt='% .16e')
    return Iq


args = parser.parse_args()

if __name__ == "__main__":
    data1 = np.loadtxt(args.file1)
    data2 = np.loadtxt(args.file2)
    # data1 = np.loadtxt('/Users/wyf/Desktop/sbw/version_manager/auto_fit_902/test/input/data1.dat')
    # data2 = np.loadtxt('/Users/wyf/Desktop/sbw/version_manager/auto_fit_902/test/input/data2.dat')
    pdb = np.vstack((data1, data2))
    # m = Molecule('test/exp/target.pdb')
    # print(len(m.data))
    # cal_intensity(m.data, 'test/exp/out_iq.dat')
    intensity_data = cal_intensity(pdb, out_iq_path=args.output)[:, :2]
    # target_data = np.loadtxt('/Users/wyf/Desktop/sbw/version_manager/auto_fit_902/test/t_iq_chi/target_iq0.dat')[:, :2]
    # interpolate
    # 内插
    target_data = np.loadtxt(args.target)[:, :2]
    td0 = target_data[:, 0]
    td1 = target_data[:, 1]
    # f = interpolate.interp1d(td0, td1)
    interpolate_x = np.around(intensity_data[:, 0], 3)
    index = len(interpolate_x) - 1
    # 去掉不在范围内的x坐标
    while interpolate_x[index] not in td0:
        intensity_data = np.delete(intensity_data, index, axis=0)
        interpolate_x = np.delete(interpolate_x, index, axis=0)
        index -= 1
    target_inter_data = np.array([[x, td1[np.where(td0 == x)[0][0]]] for x in interpolate_x if x in td0])
    # 卡方
    chi_square = np.sum(np.square(intensity_data[:, 1] - target_inter_data[:, 1]) / target_inter_data[:, 1])
    print(str(chi_square))
    one = np.ones(len(target_inter_data))
    # print(one)
    # 归一化
    curr_score = np.sum(np.square(intensity_data[:, 1] / target_inter_data[:, 1] - one)) / len(target_inter_data)

    # np.savetxt('test/iq_output/score.dat', curr_score, delimiter=' ', fmt='% .16e')
    # print('diff: ' + str(diff))
    # 方差
    # curr_score = np.sqrt(np.sum(np.square(target_inter_data - intensity_data))) / len(target_inter_data)
    print(str(curr_score))
