from FdtdApi import test_pattern_rects
from ep_utils import GetSamples
from FdtdApi import NpEncoder
from FdtdApi import FDTDModel
from FdtdApi import AnalysisSamples
import glob
import random


import json
import os
import numpy as np
def cal_spectrum(para, path='D:/sxy/optim-script/Structure pirture accruate/', file_name='para_epoch'):

    #计算光谱

    # import random
    # print('-----------------------\n-----------------------\n epoch ' + str(1) + ' start\n-----------------------\n-----------------------\n')
    para_dic = {}
    getSampleTest = GetSamples(step=1)
    # para = getSampleTest.get_random_pra_test()
    # print('-----------------------\n------------------------\n')
    print('structure parameters:', para)

    #生成结构图案

    data_t = test_pattern_rects(para)
    #     print(data_t['Exx_real'][100])#
    #     print(data_t['Exx_imag'][100])#
    #     print(data_t['Eyx_real'][100])#
    #     print(data_t['Eyx_imag'][100])#
    #     print(data_t['Eyy_real'][100])#
    #     print(data_t['Eyy_imag'][100])#
    #     para_dic['a'] = data_t['Exx_real'][100]
    #     para_dic['b'] = data_t['Exx_imag'][100]
    #     para_dic['c'] = data_t['Eyx_real'][100]
    #     para_dic['d'] = data_t['Eyx_imag'][100]
    #     para_dic['e'] = data_t['Eyy_real'][100]
    #     para_dic['f'] = data_t['Eyy_imag'][100]
    if not os.path.exists(path + file_name + '.json'):
        js = json.dumps(data_t, cls=NpEncoder)
        file = open(path + file_name + '.json', 'w')
        file.write(js)
        # path + file_name + '.json','r')
    # data_t = np.loads(path[0].replace('json','npy'))
    # data_t = json.dumps(data_json)

    asa = AnalysisSamples(r'D:\sxy\optim-script\fdtd')

    # path = 'D:/sxy/optim-script/Structure pirture accruate/'
    # dir_name = 'epoch' + str(1) + '.png'
    pic_path = path + file_name + '.png'
    # asa.draw_ep_stru(data_t,pic_path)

    #画图
    asa.draw_ep_stru_phase_amplitude(data_t, pic_path)

    # print('-----------------------\n-----------------------\n epoch ' + str(1) + ' end\n-----------------------\n-----------------------\n')
    # [112, 48, 51, 147, 52, 49, 120, 93, 80, 76]


# 使用结构画图，不用跑仿真
def draw_pattern(data_t, path='D:/sxy/optim-script/Structure pirture accruate/', file_name='para_epoch'):
    # import random
    # print('-----------------------\n-----------------------\n epoch ' + str(1) + ' start\n-----------------------\n-----------------------\n')
    para_dic = {}
    getSampleTest = GetSamples(step=1)
    # print('-----------------------\n------------------------\n')
    #    print('structure parameters:',para)
    #     print(data_t['Exx_real'][100])#
    #     print(data_t['Exx_imag'][100])#
    #     print(data_t['Eyx_real'][100])#
    #     print(data_t['Eyx_imag'][100])#
    #     print(data_t['Eyy_real'][100])#
    #     print(data_t['Eyy_imag'][100])#
    #     para_dic['a'] = data_t['Exx_real'][100]
    #     para_dic['b'] = data_t['Exx_imag'][100]
    #     para_dic['c'] = data_t['Eyx_real'][100]
    #     para_dic['d'] = data_t['Eyx_imag'][100]
    #     para_dic['e'] = data_t['Eyy_real'][100]
    #     para_dic['f'] = data_t['Eyy_imag'][100]
    if not os.path.exists(path + file_name + '.json'):
        js = json.dumps(data_t, cls=NpEncoder)
        file = open(path + file_name + '.json', 'w')
        file.write(js)
        # path + file_name + '.json','r')
    # data_t = np.loads(path[0].replace('json','npy'))
    # data_t = json.dumps(data_json)

    asa = AnalysisSamples(r'D:\sxy\optim-script\fdtd')

    # path = 'D:/sxy/optim-script/Structure pirture accruate/'
    # dir_name = 'epoch' + str(1) + '.png'
    pic_path = path + file_name + '.png'
    # asa.draw_ep_stru(data_t,pic_path)
    # asa.draw_ep_stru_phase_amplitude_detail(data_t,pic_path)
    asa.draw_ep_stru_phase_amplitude(data_t, pic_path)

    # print('-----------------------\n-----------------------\n epoch ' + str(1) + ' end\n-----------------------\n-----------------------\n')
    # [112, 48, 51, 147, 52, 49, 120, 93, 80, 76]

# 产生随机参数的函数
def get_para_random_pra_test(para, step, random_idx):
    # para 参数
    # step 每次遍历的跳数
    # random-idx 遍历的参数空间

    # old version
    #         pra1 = random.randint(30, 160)
    #         pra3 = random.randint(40, 80)
    #         pra4 = random.randint(80, 370-pra1)

    #         pra7 = random.randint(30, 150)
    #         pra8 = random.randint(40, 340-pra7)
    # 随机生成一组参数[pra1-pra10], 满足三个约束条件,note: 左闭右开
    # 对于已经找到的相近结构+- 5 参数
    # [112, 48, 52, 152, 48, 48, 112, 92, 80, 76]
    pra1 = random.randrange(para[0] - random_idx * step, para[0] + (random_idx + 1) * step, step)
    pra3 = random.randrange(para[2] - random_idx * step, para[2] + (random_idx + 1) * step, step)
    pra4 = random.randrange(para[3] - random_idx * step, para[3] + (random_idx + 1) * step, step)

    pra7 = random.randrange(para[6] - random_idx * step, para[6] + (random_idx + 1) * step, step)
    pra8 = random.randrange(para[7] - random_idx * step, para[7] + (random_idx + 1) * step, step)

    index = True

    while index:
        pra10 = 76
        pra9 = random.randrange(para[8] - random_idx * step, para[8] + (random_idx + 1) * step, step)
        pra6 = random.randrange(para[5] - random_idx * step, para[5] + (random_idx + 1) * step, step)
        pra5 = random.randrange(para[4] - random_idx * step, para[4] + (random_idx + 1) * step, step)
        pra2 = random.randrange(para[1] - random_idx * step, para[1] + (random_idx + 1) * step, step)
        if pra10 + pra9 + pra6 + pra5 + pra2 <= 370:
            return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10]
        else:
            index = True

# [154, 48, 52, 152, 48, 48, 154, 92, 80, 88]


if __name__ == '__main__':
    # random swap para space
    # 随机生成参数
    import datetime
    import os
    from tkinter import _flatten

    data_para = np.loadtxt('16 gap data 3 new.txt')
    for para_ran in data_para:

        # 路径
        path = 'D:/sxy/optim-script/Corner data/'
        # 添加当前时间构建文件夹
        today = datetime.date.today()
        dir_gs = os.path.join(path, "gen{}".format(today))



        # mesh = 4
        # 随机参数
        # para_ran = get_para_random_pra_test(para, mesh, 5)
        # 构造文件名
        file_name = str(para_ran)

        file_dir = os.path.join(dir_gs, file_name + '.json')
        #     file_dir = path + file_name + '.json'

        if not os.path.exists(dir_gs):
            os.makedirs(dir_gs)
        if not os.path.exists(file_dir):
            cal_spectrum(para=para_ran, path=dir_gs + '/', file_name=file_name)