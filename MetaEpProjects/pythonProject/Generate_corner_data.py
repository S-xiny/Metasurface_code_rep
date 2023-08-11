import numpy as np

# para 8 = 154 solid, para 10 = 88 solid para1 stand ralative position of para 8

def generate_corner_data(pra = np.array([124, 48, 52, 152, 48, 48, 124, 92, 80, 88]),
                         ran = 4,
                         data_num = 2,
                         num_dims = 8,
                         change_para = np.array([1, 2, 3, 4, 5, 6, 8, 9]) - 1 ,
                         save_txt = '4 gap data.txt'):

# para:

    # num_dims  int; number of dimensions   must left equal 8

    # pra  int,1 dimension ndarray;  parameter of structure

    # ran   int;  Neighborhood of pra ,range of swap space.             attention: if want 8 12 16 change data, change ran to 8 12 16

    # data_num  int ;  data num in linspace range , wanted data num

    # change_para  int, 1 dimension ndarray;   para wanted change, the subscript of pra, if want change para1 please give number 0.


# function:
    # generate data of high demension space corner data
    # write data in a txt by path save_txt

# return:

# generated data  int 2 dimension ndarray; wanted all data of corner data.

#

    # Generate all possible combinations of integers within the specified range for each dimension
    data = []
    for i in np.linspace(pra[change_para[0]] - ran, pra[change_para[0]] + ran, data_num):
        if num_dims == 1:
            data.append(np.array([i]))
        else:
            for j in np.linspace(pra[change_para[1]] - ran, pra[change_para[1]] + ran, data_num):
                if num_dims == 2:
                    data.append(np.array([i, j]))
                else:
                    for k in np.linspace(pra[change_para[2]] - ran, pra[change_para[2]] + ran, data_num):
                        if num_dims == 3:
                            data.append(np.array([i, j, k]))
                        else:
                            for l in np.linspace(pra[change_para[3]] - ran, pra[change_para[3]] + ran, data_num):
                                if num_dims == 4:
                                    data.append(np.array([i, j, k, l]))
                                else:
                                    for m in np.linspace(pra[change_para[4]] - ran, pra[change_para[4]] + ran,
                                                         data_num):
                                        if num_dims == 5:
                                            data.append(np.array([i, j, k, l, m]))
                                        else:
                                            for n in np.linspace(pra[change_para[5]] - ran, pra[change_para[5]] + ran,
                                                                 data_num):
                                                if num_dims == 6:
                                                    data.append(np.array([i, j, k, l, m, n, 124]))
                                                else:
                                                    for o in np.linspace(pra[change_para[6]] - ran,
                                                                         pra[change_para[6]] + ran, data_num):
                                                        if num_dims == 7:
                                                            data.append(np.array([i, j, k, l, m,n, 124, o]))
                                                        else:
                                                            for p in np.linspace(pra[change_para[7]] - ran,
                                                                                 pra[change_para[7]] + ran, data_num):
                                                                data.append(np.array([i, j, k, l, m, n, 124,o, p, 88]))

    # change data into ndarray
    data = np.array(data)

    # save data in txt
    np.savetxt(save_txt, data, fmt='%d')
    return data;
if __name__ == '__main__':
    ran = 8
    save_txt = str(ran) + ' gap data.txt'
    generate_corner_data(ran = ran, save_txt = save_txt)