#检测路径

import numpy as np
import os
import re
import glob
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from matplotlib.pyplot import MultipleLocator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
if __name__ == '__main__':
    def sep_and_combine_data(glob_path = 'C:/Users/89344/Desktop/Structure change under mesh 4 D2/*.json' , para_str = ['para7','para8'],
                            nanometer_idx = 87):

        path = glob.glob(glob_path)
        #路径
        path = np.array(path)
        path_per = []

    #     for i in path:
    #         with open (i, "r") as f:
    #             content = f.read()
    #             data_j = json.loads(content)
    #             np.save(i.replace('json','npy'), data_j)

        #正则表达式匹配 从文件名取出para 也可从npy文件提取
        regex = re.compile(r'\d\d+')
        regex_1 = re.compile(para_str[0] + r'-\d+')
        regex_6 = re.compile(para_str[1] + r'-\d+')
        for i in range(len(path)):
            if para_str[0] in path[i] and para_str[1] in path[i]:
                path_per.append(path[i])

        #参数向量
        para1 = []
        para6 = []
        data_dic = {}
        #旋转光的振幅和相位
        ampli_rl = []
        phi_rl = []
        ampli_ll = []
        phi_ll = []
        ampli_lr = []
        phi_lr = []
        eignvalues1 = []
        eignvalues2 = []
        ampli_eignvalue1 = []
        phi_eignvalue1 = []
        ampli_eignvalue2 = []
        phi_eignvalue2 = []
        eignvalue1_real = []
        eignvalue1_imag = []
        eignvalue2_real = []
        eignvalue2_imag = []
        a = []
        b = []
        c = []
        d = []
        e = []
        f = []
        rl_real = []
        lr_real = []
        ll_real = []
        rl_imag = []
        lr_imag = []
        ll_imag = []
        for i in path_per:
            para1.append(int(regex.findall(regex_1.findall(i)[0])[0]))
            para6.append(int(regex.findall(regex_6.findall(i)[0])[0]))
            data_t = np.load(i.replace('json','npy'),allow_pickle = True).item()
            ampli_rl.append(np.abs(data_t['r_rl_real'][nanometer_idx] + 1j * data_t['r_rl_imag'][nanometer_idx]))
            phi_rl.append(np.angle(data_t['r_rl_real'][nanometer_idx] + 1j * data_t['r_rl_imag'][nanometer_idx]))
            ampli_ll.append(np.abs(data_t['r_ll_real'][nanometer_idx] + 1j * data_t['r_ll_imag'][nanometer_idx]))
            phi_ll.append(np.angle(data_t['r_ll_real'][nanometer_idx] + 1j * data_t['r_ll_imag'][nanometer_idx]))
            ampli_lr.append(np.abs(data_t['r_lr_real'][nanometer_idx] + 1j * data_t['r_lr_imag'][nanometer_idx]))
            phi_lr.append(np.angle(data_t['r_lr_real'][nanometer_idx] + 1j * data_t['r_lr_imag'][nanometer_idx]))
            #本征值和本征态的振幅和相位
            ampli_eignvalue1.append(np.abs(data_t['eig_state_1_real'][nanometer_idx] + 1j * data_t['eig_state_1_imag'][nanometer_idx]))
            phi_eignvalue1.append(np.angle(data_t['eig_state_1_real'][nanometer_idx] + 1j * data_t['eig_state_1_imag'][nanometer_idx]))
            ampli_eignvalue2.append(np.abs(data_t['eig_state_2_real'][nanometer_idx] + 1j * data_t['eig_state_2_imag'][nanometer_idx]))
            phi_eignvalue2.append(np.angle(data_t['eig_state_2_real'][nanometer_idx] + 1j * data_t['eig_state_2_imag'][nanometer_idx]))

            eignvalue1_real.append(data_t['eig_state_1_real'][nanometer_idx])
            eignvalue1_imag.append(data_t['eig_state_1_imag'][nanometer_idx])
            eignvalue2_real.append((data_t['eig_state_2_real'][nanometer_idx]))
            eignvalue2_imag.append((data_t['eig_state_2_imag'][nanometer_idx]))

            a.append(data_t['Exx_real'])
            b.append(data_t['Exx_imag'])
            c.append(data_t['Eyx_real'])
            d.append(data_t['Eyx_imag'])
            e.append(data_t['Eyy_real'])
            f.append(data_t['Eyy_imag'])
            rl_real.append(data_t['r_rl_real'][nanometer_idx])
            lr_real.append(data_t['r_lr_real'][nanometer_idx])
            ll_real.append(data_t['r_ll_real'][nanometer_idx])
            rl_imag.append(data_t['r_rl_imag'][nanometer_idx])
            lr_imag.append(data_t['r_lr_imag'][nanometer_idx])
            ll_imag.append(data_t['r_ll_imag'][nanometer_idx])

        r_lr = np.array(lr_real) ** 2 + np.array(lr_imag) ** 2
        r_rl = np.array(rl_real) ** 2 + np.array(rl_imag) ** 2
        r_rr_ll = np.array(ll_real) ** 2 + np.array(ll_imag) ** 2
        data_dic['para1'] = np.array(para1)
        data_dic['para6'] = np.array(para6)
        data_dic['ampli_rl'] = np.array(ampli_rl)
        data_dic['phi_rl'] = np.array(phi_rl)
        data_dic['ampli_ll'] = np.array(ampli_ll)
        data_dic['phi_ll'] = np.array(phi_ll)
        data_dic['ampli_lr'] = np.array(ampli_lr)
        data_dic['phi_lr'] =np.array( phi_lr)
        data_dic['ampli_eignvalue1'] = np.array(ampli_eignvalue1)
        data_dic['phi_eignvalue1'] = np.array(phi_eignvalue1)
        data_dic['ampli_eignvalue2'] = np.array(ampli_eignvalue2)
        data_dic['phi_eignvalue2'] = np.array(phi_eignvalue2)
        data_dic['a'] = np.array(a)
        data_dic['b'] = np.array(b)
        data_dic['c'] = np.array(c)
        data_dic['d'] = np.array(d)
        data_dic['e'] = np.array(e)
        data_dic['f'] = np.array(f)
        data_dic['rl_real'] = np.array(rl_real)
        data_dic['lr_real'] = np.array(lr_real)
        data_dic['ll_real'] = np.array(ll_real)
        data_dic['rl_imag'] = np.array(rl_imag)
        data_dic['lr_imag'] = np.array(lr_imag)
        data_dic['ll_imag'] = np.array(ll_imag)
        data_dic['r_lr'] = np.array(r_lr)
        data_dic['r_rl'] =np.array( r_rl)
        data_dic['r_rr_ll'] = np.array(r_rr_ll)
        data_dic['wavelength'] = data_t['wavelength'][nanometer_idx]
        return data_dic

    data_total = sep_and_combine_data(glob_path = 'C:/Users/89344/Desktop/Structure change under mesh 4 D2/*.json',
                                      para_str = ['para7','para8'],
                                     nanometer_idx = 87)

    para_str = ['para7', 'para8']
    def plot3D_surface_interplolate(data_total, Z_axis='r_lr', color_map=None):
        # Z_zxix Z轴想表示的数据

        # function 插值之后画平面
        # grid 插值
        target_nano = data_total['wavelength']
        phi = None

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        font = {'family': 'Times New Roman', 'size': 12}
        every_line = 11
        X = np.array(data_total['para1']).reshape(-1)
        Y = np.array(data_total['para6']).reshape(-1)
        Z = np.array(data_total[Z_axis]).reshape(-1)

        xi = np.linspace(min(X), max(X))
        yi = np.linspace(min(Y), max(Y))
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata(pd.DataFrame(np.transpose([X, Y])), Z, (xi, yi), method='cubic')

        if color_map is not None:
            color_data = data_total[color_map].reshape(-1)
            color_data_inter = griddata(pd.DataFrame(np.transpose([X, Y])), color_data, (xi, yi), method='cubic')
            cmap = cm.turbo
            norm = Normalize(vmin=np.nanmin(color_data_inter), vmax=np.nanmax(color_data_inter))
            fc = cmap(norm(color_data_inter))
            phi = 'cm_' + color_map

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(xi, yi, zi, linewidth=0, antialiased=False, facecolors=(fc))
        ax.set_title('三维图')
        ax.set_xlabel(para_str[0] + '(nm)')
        ax.set_ylabel(para_str[1] + '(nm)')
        ax.set_zlabel(Z_axis)
        # ax3D.set_xlim('para2(nm)')
        # ax3D.set_ylim('para7(nm)')
        # ax.set_zlim(-0.1,0.1)
        ax.set_title(Z_axis, fontdict=font)

        cb = fig.colorbar(surf, shrink=0.3, aspect=5)

        if color_map is not None:
            cb.set_label(color_map)
            cb.mappable.set_clim(vmin=np.nanmin(color_data), vmax=np.nanmax(color_data))

        savepath = 'C:/Users/89344/Desktop/picture/2D change/' + para_str[0] + '_' + para_str[1] + '_target=' + str(
            round(target_nano, 1)) + Z_axis + phi + '_.png'

        if savepath != 'none':
            plt.savefig(savepath)
        plt.show()


    plot3D_surface_interplolate(data_total, Z_axis='ampli_rl', color_map='phi_rl')
