# -*- coding:utf-8 -*-
import os, sys
import copy
import numpy as np
from random import sample, uniform, choice
from scipy import ndimage, spatial, optimize,interpolate
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import intensity



T = 100
T_MIN = 1
FACTOR = 0.98
# K = 1000000
K = 5000
pdb_path = 'input.pdb'
# profile_path = 'target_saxs.dat'
write_path = 'output.pdb'
iq_path = 'iq.dat'
candidate_opt = {}
# 用于保存旋转前的pdbs
# pdb_old


electrons = {'H': 1,'HE': 2,'He': 2,'LI': 3,'Li': 3,'BE': 4,'Be': 4,'B': 5,'C': 6,'N': 7,'O': 8,'F': 9,'NE': 10,'Ne': 10,'NA': 11,'Na': 11,'MG': 12,'Mg': 12,'AL': 13,'Al': 13,'SI': 14,'Si': 14,'P': 15,'S': 16,'CL': 17,'Cl': 17,'AR': 18,'Ar': 18,'K': 19,'CA': 20,'Ca': 20,'SC': 21,'Sc': 21,'TI': 22,'Ti': 22,'V': 23,'CR': 24,'Cr': 24,'MN': 25,'Mn': 25,'FE': 26,'Fe': 26,'CO': 27,'Co': 27,'NI': 28,'Ni': 28,'CU': 29,'Cu': 29,'ZN': 30,'Zn': 30,'GA': 31,'Ga': 31,'GE': 32,'Ge': 32,'AS': 33,'As': 33,'SE': 34,'Se': 34,'Se': 34,'Se': 34,'BR': 35,'Br': 35,'KR': 36,'Kr': 36,'RB': 37,'Rb': 37,'SR': 38,'Sr': 38,'Y': 39,'ZR': 40,'Zr': 40,'NB': 41,'Nb': 41,'MO': 42,'Mo': 42,'TC': 43,'Tc': 43,'RU': 44,'Ru': 44,'RH': 45,'Rh': 45,'PD': 46,'Pd': 46,'AG': 47,'Ag': 47,'CD': 48,'Cd': 48,'IN': 49,'In': 49,'SN': 50,'Sn': 50,'SB': 51,'Sb': 51,'TE': 52,'Te': 52,'I': 53,'XE': 54,'Xe': 54,'CS': 55,'Cs': 55,'BA': 56,'Ba': 56,'LA': 57,'La': 57,'CE': 58,'Ce': 58,'PR': 59,'Pr': 59,'ND': 60,'Nd': 60,'PM': 61,'Pm': 61,'SM': 62,'Sm': 62,'EU': 63,'Eu': 63,'GD': 64,'Gd': 64,'TB': 65,'Tb': 65,'DY': 66,'Dy': 66,'HO': 67,'Ho': 67,'ER': 68,'Er': 68,'TM': 69,'Tm': 69,'YB': 70,'Yb': 70,'LU': 71,'Lu': 71,'HF': 72,'Hf': 72,'TA': 73,'Ta': 73,'W': 74,'RE': 75,'Re': 75,'OS': 76,'Os': 76,'IR': 77,'Ir': 77,'PT': 78,'Pt': 78,'AU': 79,'Au': 79,'HG': 80,'Hg': 80,'TL': 81,'Tl': 81,'PB': 82,'Pb': 82,'BI': 83,'Bi': 83,'PO': 84,'Po': 84,'AT': 85,'At': 85,'RN': 86,'Rn': 86,'FR': 87,'Fr': 87,'RA': 88,'Ra': 88,'AC': 89,'Ac': 89,'TH': 90,'Th': 90,'PA': 91,'Pa': 91,'U': 92,'NP': 93,'Np': 93,'PU': 94,'Pu': 94,'AM': 95,'Am': 95,'CM': 96,'Cm': 96,'BK': 97,'Bk': 97,'CF': 98,'Cf': 98,'ES': 99,'Es': 99,'FM': 100,'Fm': 100,'MD': 101,'Md': 101,'NO': 102,'No': 102,'LR': 103,'Lr': 103,'RF': 104,'Rf': 104,'DB': 105,'Db': 105,'SG': 106,'Sg': 106,'BH': 107,'Bh': 107,'HS': 108,'Hs': 108,'MT': 109,'Mt': 109}


class PDB(object):
    """Load pdb file."""
    def __init__(self, filename):
        self.natoms = 0
        with open(filename) as f:
            for line in f:
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue # skip other lines
                self.natoms += 1
        self.atomnum = np.zeros((self.natoms),dtype=int)
        self.atomname = np.zeros((self.natoms),dtype=np.dtype((str,3)))
        self.atomalt = np.zeros((self.natoms),dtype=np.dtype((str,1)))
        self.resname = np.zeros((self.natoms),dtype=np.dtype((str,3)))
        self.resnum = np.zeros((self.natoms),dtype=int)
        self.chain = np.zeros((self.natoms),dtype=np.dtype((str,1)))
        self.coords = np.zeros((self.natoms, 3))
        self.occupancy = np.zeros((self.natoms))
        self.b = np.zeros((self.natoms))
        self.atomtype = np.zeros((self.natoms),dtype=np.dtype((str,2)))
        self.charge = np.zeros((self.natoms),dtype=np.dtype((str,2)))
        self.nelectrons = np.zeros((self.natoms),dtype=int)
        with open(filename) as f:
            atom = 0
            for line in f:
                if line[0:4] != "ATOM" and line[0:4] != "HETA":
                    continue # skip other lines
                line = line.strip('\n')
                self.atomnum[atom] = int(line[6:11])
                self.atomname[atom] = line[13:16]
                self.atomalt[atom] = line[16]
                self.resname[atom] = line[17:20]
                self.resnum[atom] = int(line[22:26])
                self.chain[atom] = line[21]
                self.coords[atom, 0] = float(line[30:38])
                self.coords[atom, 1] = float(line[38:46])
                self.coords[atom, 2] = float(line[46:54])
                self.occupancy[atom] = float(line[54:60])
                self.b[atom] = float(line[60:66])
                atomtype = line[76:78].strip()
                if len(atomtype) == 2:
                    atomtype0 = atomtype[0].upper()
                    atomtype1 = atomtype[1].lower()
                    atomtype = atomtype0 + atomtype1
                self.atomtype[atom] = atomtype
                self.charge[atom] = line[78:80]
                self.nelectrons[atom] = electrons.get(self.atomtype[atom].upper(),6)
                atom += 1

    def write(self, filename):
        """Write PDB file format using pdb object as input."""
        records = []
        num = self.natoms // 2
        for k in range(2):
            for i in range(num):
                atomnum = '%5i' % self.atomnum[i+num*k]
                atomname = '%3s' % self.atomname[i+num*k]
                atomalt = '%1s' % self.atomalt[i+num*k]
                resnum = '%4i' % self.resnum[i+num*k]
                resname = '%3s' % self.resname[i+num*k]
                chain = '%1s' % self.chain[i+num*k]
                x = '%8.3f' % self.coords[i+num*k, 0]
                y = '%8.3f' % self.coords[i+num*k, 1]
                z = '%8.3f' % self.coords[i+num*k, 2]
                o = '% 6.2f' % self.occupancy[i+num*k]
                b = '%6.2f' % self.b[i+num*k]
                atomtype = '%2s' % self.atomtype[i+num*k]
                charge = '%2s' % self.charge[i+num*k]
                records.append(['ATOM  ' + atomnum + '  ' + atomname + ' ' + resname + ' ' + chain + resnum + '    ' + x + y + z + o + b + '          ' + atomtype + charge])
            records.append(['TER'])
        np.savetxt(filename, records, fmt='%-80s')

    def copy(self, filename, multiple=2):
        '''copy the molecule'''
        records = []
        for m in range(multiple):
            for i in range(self.natoms):
                atomnum = '%5i' % self.atomnum[i]
                atomname = '%3s' % self.atomname[i]
                atomalt = '%1s' % self.atomalt[i]
                resnum = '%4i' % self.resnum[i]
                resname = '%3s' % self.resname[i]
                chain = '%1s' % self.chain[i]
                x = '%8.3f' % self.coords[i, 0]
                y = '%8.3f' % self.coords[i, 1]
                z = '%8.3f' % self.coords[i, 2]
                o = '% 6.2f' % self.occupancy[i]
                b = '%6.2f' % self.b[i]
                atomtype = '%2s' % self.atomtype[i]
                charge = '%2s' % self.charge[i]
                records.append(['ATOM  ' + atomnum + '  ' + atomname + ' ' + resname + ' ' + chain + resnum + '    ' + x + y + z + o + b + '          ' + atomtype + charge])
            records.append(['ENDMDL'])
        np.savetxt(filename, records, fmt='%-80s')

    def update(self, molecule):
        atom_list = molecule.atoms
        # begin with the index of 0, molecule 0
        index = 0
        for a in atom_list:
            self.coords[index, 0] = a.xformCoord().x
            self.coords[index, 1] = a.xformCoord().y
            self.coords[index, 2] = a.xformCoord().z
            index += 1


side = 300
nsamples = 16 #n samples
resolution = 10
eps = 1e-6
threshold_int = 0.0


def mydist(mtr1, mtr2):
    '''
    calculate the distance of two matrix, use the euclidean, and no squar
    :param mtr1:
    :param mtr2:
    :return:distance array
    '''
    mtr1_row, mtr1_colum = np.shape(mtr1)
    mtr2_row, mtr2_colum = np.shape(mtr2)
    if mtr1_colum != mtr2_colum:
        print('The two array not the same column, please check it.')
        return
    dist = np.zeros((mtr1_row, mtr2_row))
    for i in range(mtr1_row):
        for j in range(mtr2_row):
            temp = mtr1[i] - mtr2[j]
            dist[i][j] = np.sum(temp*temp)
    return dist


def rotate(r, pdb):
    '''
    :param a:
    :param b:
    :param angle:
    :return:
    '''
    print('rotate one of molecules')

    num = pdb.natoms // 2
    pdb.coords[:num, :] = r.apply(pdb.coords[:num, :])


def translate(x, y, z, pdb):
    '''
    move one of the molecules
    :param x:
    :param y:
    :param z:
    :param pdb:
    :return:
    '''
    print('move one of the molecules')
    num = pdb.natoms // 2
    pdb.coords[:num, :] = pdb.coords[:num, :] + [x, y, z]


def detect_state(pdb,  cx0, cy0, cz0, cx1, cy1, cz1):
    '''
    检测两个分子是否重叠或者太远，重叠则将其拉出，太远则将其拉近
    :param pdb:
    :param cx0: 第一个分子中心坐标
    :param cy0:
    :param cz0:
    :param cx1: 第二个分子中心坐标
    :param cy1:
    :param cz1:
    :return:
    '''
    # 两个分子之间的状态以及最短距离
    mol_state, last_min_distance = position_state(pdb)
    step = 4  # 调整的幅度
    push_x, push_y, push_z = cx0 - cx1, cy0 - cy1, cz0 - cz1
    mod_length = np.sqrt(np.sum(np.square([push_x, push_y, push_z])))
    if mod_length != 0:
        push_x, push_y, push_z = push_x / mod_length + step, push_y / mod_length + step, push_z / mod_length + step
    else:
        push_x, push_y, push_z = step, 0, 0
    while mol_state == 'far':  # 隔离拉近
        [rx0, ry0, yz0] = pdb.coords[pdb.natoms // 4]
        [rx1, ry1, yz1] = pdb.coords[pdb.natoms // 2 + pdb.natoms // 4]  # 不动
        drag_x, drag_y, drag_z = rx1 - rx0, ry1 - ry0, yz1 - yz0
        translate(drag_x, drag_y, drag_z, pdb)
        mol_state, min_distance = position_state(pdb)
        if min_distance > last_min_distance:  # 说明方向反了
            translate(-drag_x * 2, -drag_y * 2, -drag_z * 2, pdb)
        last_min_distance = min_distance
    while mol_state == 'overlap':  # 重叠拉开
        translate(push_x, push_y, push_z, pdb)
        mol_state, last_min_distance = position_state(pdb)



# args contains pdb and profile_path)
def score(vect, *args):
    '''
    打分函数，目前有10个参数
    :param vect: 对分数有影响的变量列表，按顺序为：平移x,y,z,a,b,c,angle,
    :param args: 其他需要的函数，按顺序为pdb,目标数据target_dat（ndarry）,最终保存iq数据的路径out_iq_path,第二个分子中心cx1, cy1, cz1
    :return:分数
    '''
    print('calculate the error')
    # translate and rotate
    x = vect[0]
    y = vect[1]
    z = vect[2]
    # 传入角度
    a = vect[0]
    b = vect[1]
    c = vect[2]
    angle = vect[3]
    pdb = args[0]
    target_data = args[1]
    out_iq_path = args[2]
    # 方向向量的计算方式：找出中心点，再随机取出分子中的一个点
    cx0, cy0, cz0 = centroid(pdb)  # 第一个分子中心
    # rand_coord = pdb.coords[np.random.randint(0, pdb.natoms // 2)]
    # a = cx0 - rand_coord[0]
    # b = cy0 - rand_coord[1]
    # c = cz0 - rand_coord[2]
    r = R.from_quat([angle, a, b, c])
    # 第二个分子中心, 一直不变
    cx1, cy1, cz1 = args[3], args[4], args[5]
    rotate(r, pdb)
    translate(x, y, z, pdb)
    # 检测是否重叠或者太远
    detect_state(pdb, cx0, cy0, cz0, cx1, cy1, cz1)
    # 会改变pdb的必须传副本
    # pdb.write('temp.pdb')
    pdb_copy = copy.deepcopy(pdb)
    # denss方法
    intensity_data = intensity.cal_intensity(pdb_copy, out_iq_path)[:, :2]
    # interpolate
    # 内插
    td0 = target_data[:, 0]
    td1 = target_data[:, 1]
    # f = interpolate.interp1d(td0, td1)
    interpolate_x = np.around(intensity_data[:, 0], 3)
    index = len(interpolate_x)-1
    # 去掉不在范围内的x坐标
    while interpolate_x[index] not in td0:
        intensity_data = np.delete(intensity_data, index, axis=0)
        interpolate_x = np.delete(interpolate_x, index, axis=0)
        index -= 1
    target_inter_data = np.array([[x, td1[np.where(td0 == x)[0][0]]] for x in interpolate_x if x in td0])
    curr_score = np.sqrt(np.sum(np.square(target_inter_data - intensity_data)))
    print('current score is:' + str(curr_score))
    return curr_score


def centroid(pdb, index=0):
    '''
    centroid of one molecule, 默认计算第一个分子的中心
    :param pdb:
    :return:
    '''
    num = pdb.natoms // 2
    cx = np.mean(pdb.coords[num*index:num+num*index, 0])
    cy = np.mean(pdb.coords[num*index:num+num*index, 1])
    cz = np.mean(pdb.coords[num*index:num+num*index, 2])
    return cx, cy, cz

# ,, cx0, cy0, cz0, cx1, cy1, cz1
def position_state(pdb):
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
    state = ['overlap', 'far', 'normal']
    num = pdb.natoms // 2
    dist_min_threshold = 0.6
    dist_max_threshold = 4
    # 计算两个分子中每个原子的距离
    arr0 = pdb.coords[:num, :]
    arr1 = pdb.coords[num:, :]
    dist_map = spatial.distance.cdist(arr0, arr1)
    min_distance = np.min(dist_map)
    print('min distance:' + str(min_distance))
    if min_distance < dist_min_threshold:
        print('The molecules are overlap')
        return state[0], min_distance      # overlap
    if min_distance > dist_max_threshold:
        print('The molecules are too far')
        return state[1], min_distance      # far
    print('Normal distance')
    return state[2], min_distance         # normal


def data_display(profile_path, iq_path):
    '''
    显示拟合曲线
    :param profile_path: 目标IQ文件路径
    :param iq_path: 预测IQ文件路径
    :return: void
    '''
    target_data = np.loadtxt(profile_path)[:, :2]
    iq_data = np.loadtxt(iq_path)[:, :2]
    plt.plot(target_data[:, 0],target_data[:, 1])
    plt.plot(iq_data[:, 0], iq_data[:, 1])
    plt.show()


# implementation of simulated annealing algorithm
def simulated_annealing(error, threshold, itera, T, T_MIN, factor, K, pdb, target_dat, out_iq_path):
    '''
    implementation of simulated annealing algorithm
    :param error:
    :param threshold:
    :param iter:
    :param t:
    :param t_min:
    :param factor:
    :return: error
    '''
    global candidate_opt
    for i in range(itera):
        x = uniform(-3, 3)
        y = uniform(-3, 3)
        z = uniform(-3, 3)
        a = uniform(-1, 1)
        b = uniform(-1, 1)
        c = uniform(-1, 1)
        angle = uniform(0, 90)
        # 对分数有影响的变量列表，按顺序为：平移x,y,z,angle,
        vec = [x, y, z, a, b, c, angle]
        # 其他需要的函数，按顺序为pdb, 目标数据target_dat（ndarry）, 最终保存iq数据的路径out_iq_path, 第二个分子中心cx1, cy1, cz1
        cx1, cy1, cz1 = centroid(pdb, 1)
        args = [pdb, target_dat, out_iq_path, cx1, cy1, cz1]
        curr_score = score(vec, *args)
        T = T * factor
        # the algorithm stop
        if curr_score < threshold:
            print('error is low to accept')
            return curr_score, T
        if curr_score < error:
            # accept immediatelly 平移旋转生效
            print('Accepted the change')
            error = curr_score
        else:
            # move(-x, -y, -z, pdb)
            # accept within a probability
            rand_p = uniform(0, 1)
            print('test:::::'+str((error - curr_score) / (K*T)))
            p = np.exp((error - curr_score) / (K*T))
            print('generate p:' + str(p) + '\trand p:' + str(rand_p))
            if T > T_MIN and p > rand_p:
                # accept immediatelly
                print('Accepted the change in a probability')
                # 记录局部最优点
                if not candidate_opt:
                    candidate_opt[error] = copy.deepcopy(pdb)
                else:
                    error_old = list(candidate_opt.keys())[0]
                    if error < error_old:
                        candidate_opt.pop(error_old)
                        candidate_opt[error] = copy.deepcopy(pdb)
                error = curr_score
            else:
                # refuse to accept the change 旋转无效
                r = R.from_quat([angle, a, b, c])
                r = r.inv()
                rotate(r, pdb)
                translate(-x, -y, -z, pdb)
                print('Refuse to accepted the change')
    return error, T


def handle(pdb, T, T_MIN, FACTOR, K, target_dat, out_iq_path, output):
    '''
    处理函数，在函数中调用模拟退火方法
    :param pdb:
    :param T:
    :param T_MIN:
    :param FACTOR:
    :param K:
    :param target_dat:
    :param out_iq_path:
    :param output:
    :return:
    '''
    global candidate_opt
    error = float('inf')  # the maximum of float
    threshold = 1000
    itera = 4
    while error > threshold and T > T_MIN:
    # for i in range(5):
        error, T = simulated_annealing(error, threshold, itera, T, T_MIN, FACTOR, K, pdb,target_dat, out_iq_path)
        print('\nerror:',error, '\tT:', T)
    if candidate_opt:
        score_candi = list(candidate_opt.keys())[0]
        if score_candi < error:
            pdb = candidate_opt[score_candi]
        candidate_opt.clear()
    pdb.write(output)
    print('end of handle')


def visual_test(pdb):
    '''
    画图
    :param pdb:
    :return:
    '''
    for i in range(10):
        x = uniform(-3, 3)
        y = uniform(-3, 3)
        z = uniform(-3, 3)
        a = uniform(-1, 1)
        b = uniform(-1, 1)
        while np.power(a, 2) + np.power(b, 2) > 1:
            a = uniform(-1, 1)
            b = uniform(-1, 1)
        ctemp = np.sqrt(1 - np.power(a, 2) - np.power(b, 2))
        c = choice([ctemp, -ctemp])
        angle = uniform(0, 90)
        r = R.from_quat([angle, a, b, c])
        rotate(r, pdb)
        translate(x, y, z, pdb)
        print('\nx:', x, '\t', y, '\t', z, '\t', a, '\t', b, '\t', angle)
        Iq = intensity.cal_intensity(pdb)
        np.savetxt('iq'+str(i)+'.dat', Iq, delimiter=' ', fmt='% .16e')
        pdb.write('t' + str(i) + '.pdb')


def correct_test(test_path):
    input_pdb = 'input.pdb'
    output = test_path
    pdb = PDB(input_pdb)
    # pdb.copy('input.pdb')
    # pdb = PDB('input.pdb')
    for i in range(10):
        x = uniform(-10, 10)
        y = uniform(-10, 10)
        z = uniform(-10, 10)
        a = uniform(-10, 10)
        b = uniform(-10, 10)
        while np.power(a, 2) + np.power(b, 2) > 1:
            a = uniform(-1, 1)
            b = uniform(-1, 1)
        ctemp = np.sqrt(1 - np.power(a, 2) - np.power(b, 2))
        c = choice([ctemp, -ctemp])
        angle = uniform(0, 90)
        r = R.from_quat([angle, a, b, c])
        print('\nx:', x, '\t', y, '\t', z, '\t', a, '\t', b, '\t', angle)
        rotate(r, pdb)
        translate(x, y, z, pdb)
        cx0, cy0, cz0 = centroid(pdb)
        cx1, cy1, cz1 = centroid(pdb, 1)
        detect_state(pdb,cx0, cy0, cz0, cx1, cy1, cz1)
        Iq = intensity.cal_intensity(pdb)
        np.savetxt(output + 'iq' + str(i) + '.dat', Iq, delimiter=' ', fmt='% .16e')
        pdb.write(output + 't' + str(i) + '.pdb')

def cal_int(prefix):
    for i in range(5):
        pdb_path = prefix+'t'+str(i)+'.pdb'
        pdb=PDB(pdb_path)
        output = prefix+'iq'+str(i)+'.dat'
        intensity.cal_intensity(pdb, output)


def test():
    # correct_test('test/')
    prefix = 'test/'
    for i in range(7):
        pdb_path = 'input.pdb'
        pdb = PDB(pdb_path)
        profile_path = prefix + 't_iq_chi/' + 'target_iq' + str(i) + '.dat'
        target_dat = np.loadtxt(profile_path)[:, :2]
        output = prefix + 'pdb/' + 'output' + str(i) + '.pdb'
        out_iq_path = prefix + 'iq_output/' + 'out_iq_' + str(i) + '.dat'
        handle(pdb, T, T_MIN, FACTOR, K, target_dat, out_iq_path, output)


# use scipy annealing
# if __name__ == "__main__":
#     print('begin')
#     path_pre = 'correct_test/dat/test/'
#     pdb_path = 'test/input.pdb'
#     for i in range(1):
#         t = time.time()
#         # profile_path = path_pre + str(i) + '.dat'
#         profile_path = '/Users/wyf/Desktop/sbw/auto_fit822/correct_test/dat/t0_t0_saxs.dat'
#         output_pdb = path_pre + str(i) + 'out.pdb'
#         out_iq_path = path_pre + str(i) + 'out.dat'
#         pdb = PDB(pdb_path)
#         profile_data = np.loadtxt(profile_path)[:, :2]
#         x1 = [-10]*3
#         x2 = [10]*3
#         a1 = [-1]*3
#         a2 = [1]*3
#         ag1 = [0]
#         ag2 = [90]
#         # parameter list
#         paras = list(zip(x1, x2))+list(zip(a1, a2)+list(zip(ag1, ag2))
#         # 计算固定分子的中心, 中心一直不变
#         cx, cy, cz = centroid(pdb, 1)
#         add_args = [pdb, profile_data, out_iq_path, cx, cy, cz]
#         ret = optimize.dual_annealing(score, bounds=paras, args=add_args, initial_temp=100.)
#         print('x:' + str(ret.x) + '\tfun:' + str(ret.fun))
#         pdb.write(output_pdb)
#         data_display(profile_path, out_iq_path)
#         print('Time of one iteration is:' + str(time.time() - t))
#     print('end')



# use mine
if __name__ == "__main__":
    print('begin')
    test()
    print('end')
