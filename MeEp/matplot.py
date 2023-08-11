#插值增加平滑度
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mayavi.mlab import *
from mayavi import mlab
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
import scipy
from scipy.interpolate import griddata
from scipy import interpolate
from matplotlib import cm

from matplotlib.pyplot import MultipleLocator
if __name__ == '__main__':
    path = glob.glob('C:/Users/89344/Desktop/Structure change under mesh 4 D2/*.json')

    # 路径
    path = np.array(path)
    path_per = []

    for i in path:
        with open(i, "r") as f:
            content = f.read()
            data_j = json.loads(content)
            np.save(i.replace('json', 'npy'), data_j)
    #
    para_str = ['para1', 'para7']

    # 正则表达式匹配
    regex = re.compile(r'\d\d+')
    regex_1 = re.compile(para_str[0] + r'-\d+')
    regex_6 = re.compile(para_str[1] + r'-\d+')
    for i in range(len(path)):
        if para_str[0] in path[i] and para_str[1] in path[i]:
            path_per.append(path[i])

    # 参数向量
    para1 = []
    para6 = []

    # 旋转光的振幅和相位
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
    nanometer_idx = 87
    for i in path_per:
        para1.append(int(regex.findall(regex_1.findall(i)[0])[0]))
        para6.append(int(regex.findall(regex_6.findall(i)[0])[0]))
        data_t = np.load(i.replace('json', 'npy'), allow_pickle=True).item()
        ampli_rl.append(np.abs(data_t['r_rl_real'][nanometer_idx] + 1j * data_t['r_rl_imag'][nanometer_idx]))
        phi_rl.append(np.angle(data_t['r_rl_real'][nanometer_idx] + 1j * data_t['r_rl_imag'][nanometer_idx]))
        ampli_ll.append(np.abs(data_t['r_ll_real'][nanometer_idx] + 1j * data_t['r_ll_imag'][nanometer_idx]))
        phi_ll.append(np.angle(data_t['r_ll_real'][nanometer_idx] + 1j * data_t['r_ll_imag'][nanometer_idx]))
        ampli_lr.append(np.abs(data_t['r_lr_real'][nanometer_idx] + 1j * data_t['r_lr_imag'][nanometer_idx]))
        phi_lr.append(np.angle(data_t['r_lr_real'][nanometer_idx] + 1j * data_t['r_lr_imag'][nanometer_idx]))
        # 本征值和本征态的振幅和相位
        ampli_eignvalue1.append(
            np.abs(data_t['eig_state_1_real'][nanometer_idx] + 1j * data_t['eig_state_1_imag'][nanometer_idx]))
        phi_eignvalue1.append(
            np.angle(data_t['eig_state_1_real'][nanometer_idx] + 1j * data_t['eig_state_1_imag'][nanometer_idx]))
        ampli_eignvalue2.append(
            np.abs(data_t['eig_state_2_real'][nanometer_idx] + 1j * data_t['eig_state_2_imag'][nanometer_idx]))
        phi_eignvalue2.append(
            np.angle(data_t['eig_state_2_real'][nanometer_idx] + 1j * data_t['eig_state_2_imag'][nanometer_idx]))

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




    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'size': 12}
    font_legend = {'family': 'Times New Roman', 'size': 10}

    every_line = 11

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font = {'family': 'Times New Roman', 'size': 12}
    font_legend = {'family': 'Times New Roman', 'size': 10}
    fig = plt.figure(figsize=(12, 8))
    ax3D = Axes3D(fig,auto_add_to_figure=False)
    every_line = 11
    fig.add_axes(ax3D)

    x0 = np.array(list(set(para1)))
    y0 = np.array(list(set(para6)))
    x0.sort()
    y0.sort()
    X0, Y0 = np.meshgrid(x0,y0)
    Z0 = np.array(r_lr).reshape(every_line,-1)
    Z1 = np.array(r_rl).reshape(every_line,-1)
    Z2 = np.array(r_rr_ll).reshape(every_line,-1)
    #插值
    func0 = interpolate.interp2d(X0.T.ravel(),Y0.T.ravel(),Z0.ravel(), kind = 'linear')
    xnew = np.linspace(x0.min(), x0.max(),every_line**2)
    ynew = np.linspace(y0.min(), y0.max(),every_line**2)
    Znew0 = func0(xnew.ravel(), ynew.ravel())

    func1 = interpolate.interp2d(X0.T.ravel(),Y0.T.ravel(),Z1.ravel(), kind = 'linear')
    Znew1= func1(xnew.ravel(), ynew.ravel())

    func2 = interpolate.interp2d(X0.T.ravel(),Y0.T.ravel(),Z2.ravel(), kind = 'linear')
    Znew2= func2(xnew.ravel(), ynew.ravel())


    Xnew, Ynew= np.meshgrid(xnew,ynew)
    Xnew = Xnew.T
    Ynew = Ynew.T
    #Z = np.array(ampli_rl).reshape(every_line,-1)
    surf = ax3D.plot_surface(Xnew, Ynew, Znew0.T,
                            rstride=1,  # rstride（row）指定行的跨度
                            cstride=1,  # cstride(column)指定列的跨度
                            linewidth=0.5,
                            antialiased=True,
                            shade = True,
                            cmap = 'coolwarm',
                            label = 'r_lr')
    surf = ax3D.plot_surface(Xnew, Ynew, Znew1.T,
                            rstride=1,  # rstride（row）指定行的跨度
                            cstride=1,  # cstride(column)指定列的跨度
                            linewidth=0.5,
                            antialiased=True,
                            shade = True,
                            cmap = 'BuPu',
                            label = 'r_rl')
    surf = ax3D.plot_surface(Xnew, Ynew, Znew2.T,
                            rstride=1,  # rstride（row）指定行的跨度
                            cstride=1,  # cstride(column)指定列的跨度
                            linewidth=0.5,
                            antialiased=True,
                            shade = True,
                            cmap = 'spring',
                            label = 'r_rr_ll')
    ax3D.set_xlabel(para_str[0]+ '(nm)')
    ax3D.set_ylabel(para_str[1]+ '(nm)')
    ax3D.set_zlabel('Amplitude')
    # ax3D.set_xlim('para2(nm)')
    # ax3D.set_ylim('para7(nm)')
    #ax3D.set_zlim(-0.6,0.6)
    ax3D.set_title('r_rl', fontdict=font)
    cb = plt.colorbar(surf,shrink=0.3, aspect=5)

    plt.show()