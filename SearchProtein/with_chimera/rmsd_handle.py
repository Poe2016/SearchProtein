# -*- coding:utf-8 -*-
import chimera
from chimera import runCommand
import Midas  # 用于chimera中写文件保存pdb
import os, sys
import copy
import numpy as np
from random import sample, uniform, choice, randint
import matplotlib.pyplot as plt
import time

# 用于保存局部最优解
candidate_opt = {}
# 用于保存历史score分数，用于最后的作图
score_list = []
# 用于保存历史差异
diff_list = []

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
        # self.atomnum = np.zeros((self.natoms), dtype=int)
        # self.atomname = np.zeros((self.natoms), dtype=np.dtype((str, 3)))
        # self.atomalt = np.zeros((self.natoms), dtype=np.dtype((str, 1)))
        # self.resname = np.zeros((self.natoms), dtype=np.dtype((str, 3)))
        # self.resnum = np.zeros((self.natoms), dtype=int)
        # self.chain = np.zeros((self.natoms), dtype=np.dtype((str, 1)))
        self.coords = np.zeros((self.natoms, 3))
        # self.occupancy = np.zeros((self.natoms))
        # self.b = np.zeros((self.natoms))
        # self.atomtype = np.zeros((self.natoms), dtype=np.dtype((str, 2)))
        # self.charge = np.zeros((self.natoms), dtype=np.dtype((str, 2)))
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
        data = np.hstack((self.coords, self.nelectrons.reshape(-1, 1)))
        np.savetxt(self.data_path, data, fmt='%-40s')

    def update(self, molecule):
        '''
        需要根据chimera中的分子molecule来更新文件中的分子坐标
        :param molecule:
        :return:
        '''
        atoms = molecule.atoms
        natom = len(atoms)
        for i in range(natom):
            self.coords[i, 0] = atoms[i].xformCoord()[0]
            self.coords[i, 1] = atoms[i].xformCoord()[1]
            self.coords[i, 2] = atoms[i].xformCoord()[2]
        data = np.hstack((self.coords, self.nelectrons.reshape(-1, 1)))
        np.savetxt(self.data_path, data, fmt='%-40s')

    def centroid(self):
        '''
        计算分子中心
        :param pdb:
        :return:
        '''
        cx = np.mean(self.coords[:, 0])
        cy = np.mean(self.coords[:, 1])
        cz = np.mean(self.coords[:, 2])
        return cx, cy, cz

    def write(self):
        '''
        用于最终的保存结果
        :return:
        '''
        with open(self.filename, 'w') as f:
            data = ''
            index = 0
            for line in f:
                if line[:5] != 'ATOM':
                    data += line
                    continue
                else:
                    line = line[:30] + str(self.coords[index, 0]).rjust(8) + str(self.coords[index, 1]).rjust(8) + str(
                        self.coords[index, 2]).rjust(8) + line[54:]
                    data += line
                    index += 1
            f.write(data)


class COMP(object):
    '''
    复合物类，复合物是指蛋白质分子的复合物，包括蛋白质分子个数以及蛋白质分子的详细信息
    '''

    def __init__(self, **data_dict):
        '''
        初始化复合物，复合物由给定的输入pdb文件确定，每个pdb文件代表一个分子
        :param filelist:输入文件列表
        '''
        filelist = list(data_dict.keys())
        self.nmolecules = len(filelist)
        self.molecules = []
        for filename in filelist:
            m = Molecule(filename, data_dict[filename])
            self.molecules.append(m)


def cent_chi(molecule):
    coord = np.zeros(3)
    atoms = molecule.atoms
    for i in range(len(atoms)):
        coord[0] += atoms[i].xformCoord()[0]
        coord[1] += atoms[i].xformCoord()[1]
        coord[2] += atoms[i].xformCoord()[2]
    cx = coord[0] / len(atoms)
    cy = coord[1] / len(atoms)
    cz = coord[2] / len(atoms)
    return cx, cy, cz


def rotate(a, b, c, angle, molecule):
    axis = chimera.Vector(a, b, c)
    xf = chimera.Xform.rotation(axis, angle)
    molecule.openState.globalXform(xf)


def translate(x, y, z, molecule):
    '''
    from chimera import runCommand as rc
    rc('rmsd #0 #1')
    m=chimera.openModels.list()[0]
    xf = chimera.Xform.translation(chimera.Vector(10,0,0))
    m.openState.globalXform(xf)

    import io
    f=io.open('/Users/wyf/Desktop/sbw/version_manager/auto_fit_902/tt.dat','w',encoding='utf-8')
    old = sys.stdout
    sys.stdout=f
    :param x:
    :param y:
    :param z:
    :param molecule:
    :return:
    '''
    xf = chimera.Xform.translation(chimera.Vector(x, y, z))
    molecule.openState.globalXform(xf)


def detect_state(mol, mol_fixed, mol_chi, mode):
    '''
    :param mol:
    :param mol_fixed:
    :param mol_chi:
    :return:
    '''
    # 用于记录移动的向量
    translate_vect = [0, 0, 0]
    cmd = 'python /Users/wyf/Desktop/sbw/version_manager/auto_fit_905/rmsd_position.py -f1 ' + os.getcwd() + '/' + mol.data_path + ' -f2 ' + os.getcwd() + '/' + mol_fixed.data_path
    command_output = os.popen(cmd)
    result = command_output.read().split('\n')
    mol_state, last_min_distance, num_overlap= result[0], result[1], result[2]
    step = 8  # 调整的幅度
    # 两个分子中心
    cx0, cy0, cz0 = mol.centroid()
    cx1, cy1, cz1 = mol_fixed.centroid()
    push_x, push_y, push_z = cx0 - cx1 , cy0 - cy1, cz0 - cz1
    mod_length = np.sqrt(np.sum(np.square([push_x, push_y, push_z])))
    if mod_length != 0:
        push_x, push_y, push_z = push_x / mod_length * step, push_y / mod_length * step, push_z / mod_length * step
    else:
        push_x, push_y, push_z = step, 0, 0
    while mol_state == 'isolate':  # 隔离拉近
        [rx0, ry0, yz0] = mol.coords[mol.natoms // 2]
        [rx1, ry1, yz1] = mol_fixed.coords[mol_fixed.natoms // 2]  # 不动
        drag_x, drag_y, drag_z = rx1 - rx0, ry1 - ry0, yz1 - yz0
        translate(drag_x, drag_y, drag_z, mol_chi)
        mol.update(mol_chi)
        translate_vect = [translate_vect[0] + drag_x, translate_vect[1] + drag_y, translate_vect[2] + drag_z]
        print('iso ', drag_x, ' ', drag_y, ' ', drag_z)
        command_output = os.popen(
            'python /Users/wyf/Desktop/sbw/version_manager/auto_fit_905/rmsd_position.py -f1 ' + mol.data_path + ' -f2 ' + mol_fixed.data_path)
        result = command_output.read().split('\n')
        mol_state, min_distance = result[0], result[1]
        print(mol_state, ' ', min_distance)
        if min_distance > last_min_distance:  # 说明方向反了
            print('反方向----')
            translate(-drag_x * 2, -drag_y * 2, -drag_z * 2, mol_chi)
            mol.update(mol_chi)
            translate_vect = [translate_vect[0] - drag_x * 2, translate_vect[1] - drag_y * 2,
                              translate_vect[2] - drag_z * 2]
            print('----iso ', -drag_x * 2, ' ', -drag_y * 2, ' ', -drag_z * 2)

        last_min_distance = min_distance
    if mode == 'over':
        while mol_state == 'overlap':  # 重叠拉开
            translate(push_x, push_y, push_z, mol_chi)
            mol.update(mol_chi)
            translate_vect = [translate_vect[0] + push_x, translate_vect[1] + push_y, translate_vect[2] + push_z]
            print('over ', push_x, ' ', push_y, ' ', push_z)
            command_output = os.popen(
                'python /Users/wyf/Desktop/sbw/version_manager/auto_fit_905/rmsd_position.py -f1 ' + mol.data_path + ' -f2 ' + mol_fixed.data_path)
            result = command_output.read().split('\n')
            mol_state, last_min_distance = result[0], result[1]

    return translate_vect


# implementation of simulated annealing algorithm
def simulated_annealing(error, threshold, itera, T, T_MIN, factor, K, step, angle_max, target_dat, out_iq_path, molecule_chi,
                        molecule_comp, molecule_comp_fixed, rmsd_rcd, x_target, y_target, z_target ):
    '''
    :param error:
    :param threshold:
    :param itera:
    :param T:
    :param T_MIN:
    :param factor:
    :param K:
    :param target_dat:
    :param out_iq_path:
    :param molecule_chi:
    :param molecule_comp:与molecule_chi对应的COMP中的molecule
    :param molecule_comp_fixed:
    :param rmsd_rcd:
    :return:
    '''
    global candidate_opt
    global score_list
    global diff_list
    if error < threshold:
        print('error is low to accept with no loop')
        score_list.append(error)
        return error, T, step
    for i in range(itera):
        x = uniform(-step, step)
        y = uniform(-step, step)
        z = uniform(-step, step)
        angle = uniform(0, angle_max)
        # 改变之前先保存
        # molecule_chi_old = copy.deepcopy(molecule_chi)
        cx, cy, cz = molecule_comp.centroid()
        rand_num = randint(0, len(molecule_chi.atoms) - 1)
        [rx, ry, rz] = list(molecule_chi.atoms[rand_num].xformCoord())
        # rand_num2 = randint(0, len(molecule_chi.atoms) - 1)
        # while rand_num == rand_num2:
        #     rand_num2 = randint(0, len(molecule_chi.atoms) - 1)
        # [cx, cy, cz] = list(molecule_chi.atoms[rand_num2].xformCoord())
        a, b, c = rx - cx, ry - cy, rz - cz
        rotate(a, b, c, angle, molecule_chi)
        translate(x_target - cx, y_target - cy, z_target - cz, molecule_chi)
        # translate(x, y, z, molecule_chi)
        molecule_comp.update(molecule_chi)  # 修改文件中的坐标

        print(step)
        mode = 'iso'

        translate_vect = detect_state(molecule_comp, molecule_comp_fixed, molecule_chi, mode)
        # print(translate_vect)
        
        print('begin')
        cmd = 'python /Users/wyf/Desktop/sbw/version_manager/auto_fit_905/rmsd_intensity.py -f1 ' + os.getcwd() + '/' + molecule_comp.data_path + ' -f2 ' + os.getcwd() + '/' + molecule_comp_fixed.data_path + ' -t ' + os.getcwd() + '/' + target_dat + ' -o ' + os.getcwd() + '/' + out_iq_path
        command_output = os.popen(cmd)
        # 返回一个列表，按顺序对应 difference, score
        difference_score = command_output.read().split('\n')
        print('difference: ', difference_score[0])
        print('score: ', difference_score[1])
        # 输出重定向
        f = open(rmsd_rcd, 'a')
        old = sys.stdout
        sys.stdout = f
        runCommand('rmsd #0,1 #2')
        sys.stdout = old
        f.close()
        curr_score = float(difference_score[0])
        print('curr_score:' + str(curr_score))

        T = T * factor
        if step>10:
            step = step * factor
        # angle_max = angle_max * factor
        # the algorithm stop
        if curr_score < threshold:

            print('error is low to accept')
            diff_list.append(float(difference_score[0]))
            # score = different from different
            score_list.append(curr_score)
            return curr_score, T, step, angle_max
        if curr_score < error:
            # accept immediatelly 平移旋转生效
            print('Accepted the change')
            error = curr_score
        else:
            # move(-x, -y, -z, pdb)
            # accept within a probability
            rand_p = uniform(0, 1)
            print('test:::::' + str((error - curr_score) / (K * T)))
            p = np.exp((error - curr_score) / (K * T))
            print('generate p:' + str(p) + '\trand p:' + str(rand_p))
            if T > T_MIN and p > rand_p:

                # accept immediatelly
                print('Accepted the change in a probability')
                # 记录局部最优点
                # if not candidate_opt:
                #     candidate_opt[error] = molecule_chi_old
                # else:
                #     error_old = list(candidate_opt.keys())[0]
                #     if error < error_old:
                #         candidate_opt.pop(error_old)
                #         candidate_opt[error] = molecule_chi_old
                error = curr_score
            else:
                # refuse to accept the change 旋转无效
                print('-------------------------------')
                # translate(-translate_vect[0], -translate_vect[1], -translate_vect[2], molecule_chi)
                # translate(-x, -y, -z, molecule_chi)
                # rotate(a, b, c, -angle, molecule_chi)
                # molecule_comp.update(molecule_chi)
                print('Refuse to accepted the change')
        diff_list.append(float(difference_score[0]))
        score_list.append(error)

    return error, T, step, angle_max


# chimera chimera.openModels.list()[0] chimera openModls.list()


def handle():
    global candidate_opt
    global score_list
    global diff_list
    T = 1
    T_MIN = 0.0001
    FACTOR = 0.99
    K = 1
    threshold = 1000
    error = 1000000000000000
    itera = 4
    step = 100
    angle_max = 360
    # 目标pdb的实验数据文件
    target_dat = 'test/exp/out_iq.dat'
    # 计算信号强度保存文件
    out_iq_path = 'test/exp/rmsd_exp.dat'
    # 输入模型1文件
    input_f1 = 'test/exp/input1.pdb'
    # 输入模型1的数据文件，用于保存模型1的分子的原子坐标，便于计算信号强度
    data_path1 = 'test/exp/data1.dat'
    # 输入模型2文件
    input_f2 = 'test/exp/input2.pdb'
    # 输入模型2的数据文件，用于保存模型1的分子的原子坐标，便于计算信号强度
    data_path2 = 'test/exp/data2.dat'
    # rmsd 记录文件，记录chimera计算rmsd过程中产生的原始数据
    rmsd_rcd = 'test/exp/rmsd_rcd.dat'
    # 最后结果模型保存文件
    out_pdb_path = 'test/exp/result.pdb'
    # 历史分数记录文件
    out_score_path = 'test/exp/score_rcd.dat'
    # 历史差异记录文件
    out_diff_path = 'test/exp/diff_rcd.dat'
    data_dict = {input_f1: data_path1, input_f2: data_path2}
    mlist = chimera.openModels.list()
    # find_molecule()#找出chimera中的分子与分子文件之间的对应关系
    comp = COMP(**data_dict)
    molecule_chi = mlist[0]
    molecule_chi_fixed = mlist[1]
    molecule_comp = comp.molecules[1]  # 需要修改文件的对应分子
    molecule_comp_fixed = comp.molecules[0]
    molecule_comp.update(molecule_chi)
    molecule_comp_fixed.update(molecule_chi_fixed)
    molecule_chi_target = mlist[3]
    x_target, y_target, z_target = cent_chi(molecule_chi_target)
    x_curr, y_curr, z_curr = molecule_comp.centroid()
    translate(x_target - x_curr, y_target - y_curr, z_target - z_curr, molecule_chi)
    molecule_comp.update(molecule_chi)
    # cmd = 'python /Users/wyf/Desktop/sbw/version_manager/auto_fit_905/rmsd_intensity.py -f1 ' + os.getcwd() + '/' + molecule_comp.data_path + ' -f2 ' + os.getcwd() + '/' + molecule_comp_fixed.data_path + ' -t ' + os.getcwd() + '/' + target_dat + ' -o ' + os.getcwd() + '/' + out_iq_path
    # command_output = os.popen(cmd)
    # # 返回一个列表，按顺序对应 difference, score
    # difference_score = command_output.read().split('\n')
    # print('difference: ', difference_score[0])
    # print('score: ', difference_score[1])
    # # 输出重定向
    # f = open(rmsd_rcd, 'a')
    # old = sys.stdout
    # sys.stdout = f
    # runCommand('rmsd #0,1 #2')
    # sys.stdout = old
    # f.close()
    # error = difference_score[0]


    while error > threshold and T > T_MIN:
        # for i in range(5):
        error, T, step, angle_max = simulated_annealing(error, threshold, itera, T, T_MIN, FACTOR, K, step, angle_max, target_dat, out_iq_path,
                                       molecule_chi, molecule_comp, molecule_comp_fixed,
                                       rmsd_rcd, x_target, y_target, z_target)  # molecule_to_change是chimera得到的分子 molecule是comp中的分子
        print('error:', error, 'T:', T)
    # 保存最后的结果
    # molecule_comp.write()
    save_list = []
    # if candidate_opt:
    #     score_candi = list(candidate_opt.keys())[0]
    #     if score_candi < error:
    #         molecule_chi = candidate_opt[score_candi]
    #     candidate_opt.clear()
    save_list.append(molecule_chi)
    save_list.append(molecule_chi_fixed)
    Midas.write(save_list, None, out_pdb_path)
    np.savetxt(out_diff_path, diff_list)
    np.savetxt(out_score_path, score_list)
    if diff_list:
        diff_list = []
    if score_list:
        score_list = []

    print('end of handle')


if __name__ == "__main__":
    handle()
