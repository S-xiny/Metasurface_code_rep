import pandas as pd
import numpy as np

def confirm_para(para):
    for i in para:
        if i < 30:
            return False;
    if para[0] + para[3] > 370:
        return False
    if para[6] + para[7] > 370:
        return False
    if para[1] + para[4] + para[5] + para[8] + para[9] >370:
        return False
    return True


# def main():
#     # bi = [4, -4]
#     a = [[4], [-4]]
#
#     change_para = [2,3,4,5,7,8]
#
#     for i in change_para:
#         new_a = []
#         for j in a:
#             point1 = j.copy()
#             point2 = j.copy()
#             point1.append(4)
#             point2.append(-4)
#             new_a.append(point1)
#             new_a.append(point2)
#         a = new_a
#
#     list_data = []
#     initial = np.array([124, 48, 52, 152, 48, 48, 124, 92, 80, 88])
#     for k in a:
#         data = {}
#         for j in initial:
#             data['pra'+str(j)] = initial[j-1] + k[j-1]
#             list_data.append(data)
#     df = pd.DataFrame(list_data)
#     # #df.append(list_data)
#     print(df)
def main():
    # bi = [4, -4]
    gap = 24
    save_txt =  str(gap) + ' gap data 3 new.txt'
    a = [[gap], [-gap]]

    change_para = [2, 3, 4, 5, 7, 8]

    for i in range(9):
        new_a = []
        for j in a:
            point1 = j.copy()
            point2 = j.copy()
            point3 = j.copy()
            point1.append(gap)
            point2.append(-gap)
            point3.append(0)
            new_a.append(point1)
            new_a.append(point2)
            new_a.append(point3)
        a = new_a
    a = np.array(a)
    initial = np.array([124, 48, 52, 152, 48, 48, 124, 92, 80, 88])
    list_data = []
    for i in range(len(a)):
        list_data.append(initial)
    list_data = np.array(list_data)
    a[:, 0] = + 4
    a[:, 1] = - 4
    a[:, 6] = 0
    a[:, 9] = 0

    answer = a + list_data
    data_t = np.unique(answer, axis=0)
    nc_data = 0
    for iterator in range(len(data_t)):
        if confirm_para(data_t[iterator]) == False:
            nc_data +=1
            np.delete(data_t, iterator, 0)
            print(data_t[iterator])

    np.savetxt(save_txt, data_t, fmt='%d')

if __name__ == '__main__':
    main()
