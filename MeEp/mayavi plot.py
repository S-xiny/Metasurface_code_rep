# 检测路径
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

#mlab.options.backend = 'envisage'
#f = mlab.figure()
# path = glob.glob('C:/Users/89344/Desktop/Structure 2 demensition pro/Structure 2 demensition/*.json')
def sep_and_combine_data(glob_path = 'D:/sxy/optim-script/Structure change under mesh 4 D2/*.json' , para_str = ['para7','para8'],
                        nanometer_idx = 87,exist_npy = False):

    path = glob.glob(glob_path)
    #路径
    path = np.array(path)
    path_per = []
    if exist_npy == False:
        for i in path:
            with open (i, "r") as f:
                content = f.read()
                data_j = json.loads(content)
                np.save(i.replace('json','npy'), data_j)

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
        data_dic['wavelength'] = data_t['wavelength'][nanometer_idx]

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

    data_dic['unwrap_phi_rl'] = np.unwrap(np.array(phi_rl))
    data_dic['unwrap_phi_ll'] = np.unwrap(np.array(phi_ll))
    data_dic['unwrap_phi_lr'] =np.unwrap(np.array( phi_lr))
    return data_dic



if __name__ == '__main__':
    #mlab.options.backend = 'envisage'
    #f = mlab.figure()
    para_str = ['para7', 'para8']
    data_total = sep_and_combine_data(glob_path='D:/sxy/optim-script/Structure change under mesh 4 D2/*.json',
                                      para_str=['para7', 'para8'],
                                      nanometer_idx=87)
    a = data_total['a']
    b = data_total['b']
    c = data_total['c']
    d = data_total['d']
    e = data_total['e']
    f = data_total['f']
    sqt = np.sqrt(((a + 1j * b) - (e + 1j * f)) ** 2 + 4 * (c + 1j * d) ** 2)
    fwd = a + 1j * b + e + 1j * f
    lambda1 = fwd + sqt
    lambda2 = fwd - sqt
    ampli1 = np.abs(lambda1)
    phi1 = np.angle(lambda1)
    ampli2 = np.abs(lambda2)
    phi2 = np.angle(lambda2)


    # r_lr = np.array(lr_real) ** 2 + np.array(lr_imag) ** 2
    # r_rl = np.array(rl_real) ** 2 + np.array(rl_imag) ** 2
    # r_rr_ll = np.array(ll_real) ** 2 + np.array(ll_imag) ** 2
    every_line = 11
    X = np.array(data_total['para1']).reshape(every_line, -1)
    Y = np.array(data_total['para6']).reshape(every_line, -1)
    # Z1 = np.array(data_total['ampli_eignvalue1']).reshape(every_line,-1)
    Z1 = np.array(ampli1[:, 87]).reshape(every_line, -1)
    cb1 = np.array(data_total['phi_eignvalue1']).reshape(every_line, -1)
    Z2 = np.array(ampli2[:, 87]).reshape(every_line, -1)
    # Z2 = np.array(data_total['ampli_eignvalue2']).reshape(every_line,-1)
    cb2 = np.array(data_total['phi_eignvalue2']).reshape(every_line, -1)

    every_line = 11


    # x0 = np.array(para1).reshape(every_line, -1)
    # y0 = np.array(para6).reshape(every_line, -1)
    # Z0 = np.array(ampli_eignvalue1).reshape(every_line,-1)
    # Z1 = np.array(ampli_eignvalue2).reshape(every_line,-1)
    # Z2 = np.array(r_rr_ll).reshape(every_line,-1)
    # z = np.sin(np.sqrt(x0 ** 2 + y0 ** 2))
    # print(x0.shape)
    # print(y0.shape)
    # print(Z0.shape)
    # print(Z1.shape)
    # print(Z2.shape)
    #mlab.axes(x_axis_visibility = True, y_axis_visibility = True,z_axis_visibility = True)




    def set_zlim(zmin, zmax):
        # Get the current camera position and orientation
        cam_pos = mlab.view()

        # Set the limits of the Z axis
        mlab.axes(z_axis_visibility=True, ranges=[-10, 10, -10, 10, zmin, zmax])

        # Restore the camera position and orientation
        mlab.view(*cam_pos)

        # Redraw the figure
        mlab.draw()


    s1 = mlab.surf(X, Y, Z1,  warp_scale= 10,color = (1, 0, 0),colormap = 'autumn')
    s2 = mlab.surf(X, Y, Z2, warp_scale=10,color = (0.1, 0.8, 0.1), colormap = 'ocean')
   # s3 = mlab.surf(x0, y0, Z2, warp_scale=100,color = (0.1, 0.1, 0.8), colormap = 'winter')


    # s1 = mlab.surf(x0, y0, Z0,  warp_scale= 100,color = (1, 0, 0),colormap = 'autumn')
    # s2 = mlab.surf(x0, y0, Z1, warp_scale=100,color = (0.1, 0.8, 0.1), colormap = 'ocean')
    # s3 = mlab.surf(x0, y0, Z2, warp_scale=100,color = (0.1, 0.1, 0.8), colormap = 'winter')


    # extent = (output.bounds[0], output.bounds[1], output.bounds[2], output.bounds[3], output.bounds[4], output.bounds[5])
    #
    # ranges = [x0.min() , x0.max(), y0.min(),y0.max(), 0, 0.12]
    #
    mlab.axes(xlabel=para_str[0], ylabel=para_str[1], zlabel='Z')
    #mlab.colorbar(orientation='vertical', label='My Colorbar', nb_labels=5)
    cb1 = mlab.colorbar(s1 ,nb_labels = 0)
    #data = np.ones_like(x0)
    cb1.scalar_bar_representation.maximum_size = np.array([0.1, 0.8])
    cb1.scalar_bar_representation.position = np.array([0.85, 0.1])
    #
    cb1.data_range = (1, 1)  # Set the limits of the color scale to a single value
    cb1.scalar_bar_representation.position2 = np.array([0.1, 0.2])
    cb1.scalar_bar.title = 'r_lr'

    cb1.scalar_bar.title_text_property.font_size = 10
    cb1.scalar_bar.title_text_property.justification = 'right'
    cb1.scalar_bar.title_text_property.vertical_justification = 'centered'


    cb2 = mlab.colorbar(s2 ,nb_labels = 0)
    #data = np.ones_like(x0)
    cb2.scalar_bar_representation.maximum_size = np.array([0.1, 0.8])
    cb2.scalar_bar_representation.position = np.array([0.85, 0.1])
    #
    cb2.data_range = (1, 1)  # Set the limits of the color scale to a single value
    cb2.scalar_bar_representation.position2 = np.array([0.1, 0.2])
    cb2.scalar_bar.title = 'r_rl'


    # cb3 = mlab.colorbar(s3 ,nb_labels = 0)
    # #data = np.ones_like(x0)
    # cb3.scalar_bar_representation.maximum_size = np.array([0.1, 0.8])
    # cb3.scalar_bar_representation.position = np.array([0.85, 0.1])
    # #
    # cb3.data_range = (1, 1)  # Set the limits of the color scale to a single value
    # cb3.scalar_bar_representation.position2 = np.array([0.1, 0.2])
    # cb3.scalar_bar.title = 'r_rr_ll'

    # mlab.axes(xlabel=para_str[0], ylabel=para_str[1], zlabel='Z')
    # #mlab.colorbar(orientation='vertical', label='My Colorbar', nb_labels=5)
    # cb1 = mlab.colorbar(s1 ,nb_labels = 0)
    # #data = np.ones_like(x0)
    # cb1.scalar_bar_representation.maximum_size = np.array([0.1, 0.8])
    # cb1.scalar_bar_representation.position = np.array([0.85, 0.1])
    # #
    # cb1.data_range = (1, 1)  # Set the limits of the color scale to a single value
    # cb1.scalar_bar_representation.position2 = np.array([0.1, 0.5])
    # cb1.scalar_bar.label_text_property.color = (0.8, 0.1, 0.1)
    # cb1.scalar_bar.title_text_property.color = (0.8, 0.1, 0.1)
    # cb1.scalar_bar.title = 'r_lr'


    # Create a Mayavi visualization and plot the scalar field as a volume




    # cb2 = mlab.colorbar(s2 , orientation='vertical')
    # cb2.scalar_bar.title = 'My Colorbar'



    # cb3 = mlab.colorbar(s3 , orientation='vertical')
    # cb3.scalar_bar.title = 'My Colorbar'




    mlab.show()
    # Print the auto-computed warp scale factor