import os
import numpy as np
import time
import random
import matplotlib.pyplot as plt
import json
import datetime
from FdtdApi import FDTDModel
from FdtdApi import PatternRects






# 批量获得一批结构，满足预先设定约束的结构, [40*40], 每一格代表10nm
class GetSamples:
    # 参数初始化
    def __init__(self, unit=400, step=4):
        # 参数初始化
        self.unit = unit
        self.step = step

    # 随机生成一组参数[pra1-pra10], 满足三个约束条件 #原始使用
    def get_random_pra(self):
        # old version
        #         pra1 = random.randint(30, 160)
        #         pra3 = random.randint(40, 80)
        #         pra4 = random.randint(80, 370-pra1)

        #         pra7 = random.randint(30, 150)
        #         pra8 = random.randint(40, 340-pra7)
        # 随机生成一组参数[pra1-pra10], 满足三个约束条件,note: 左闭右开
        pra1 = random.randrange(30, 160, self.step)
        pra3 = random.randrange(40, 80, self.step)
        pra4 = random.randrange(80, 370 - pra1, self.step)

        pra7 = random.randrange(30, 150, self.step)
        pra8 = random.randrange(40, 340 - pra7, self.step)

        index = True

        while index:
            pra10 = random.randrange(30, 100, self.step)
            pra9 = random.randrange(40, 100, self.step)
            pra6 = random.randrange(40, 100, self.step)
            pra5 = random.randrange(40, 100, self.step)
            pra2 = random.randrange(40, 100, self.step)
            if pra10 + pra9 + pra6 + pra5 + pra2 <= 370:
                return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10]
            else:
                index = True

    # para = [112, 49, 52, 148, 53, 50, 121, 94, 80, 77]
    def get_random_pra_test(self):
        # old version
        #         pra1 = random.randint(30, 160)
        #         pra3 = random.randint(40, 80)
        #         pra4 = random.randint(80, 370-pra1)

        #         pra7 = random.randint(30, 150)
        #         pra8 = random.randint(40, 340-pra7)
        # 随机生成一组参数[pra1-pra10], 满足三个约束条件,note: 左闭右开
        # 对于已经找到的相近结构+- 5 参数
        # [112, 48, 52, 152, 48, 48, 112, 92, 80, 76]
        pra1 = random.randrange(111, 113, self.step)
        pra3 = random.randrange(51, 53, self.step)
        pra4 = random.randrange(147, min(370 - pra1, 149), self.step)

        pra7 = random.randrange(120, 122, self.step)
        pra8 = random.randrange(93, min(340 - pra7, 95), self.step)

        index = True

        while index:
            pra10 = random.randrange(76, 78, self.step)
            pra9 = random.randrange(79, 81, self.step)
            pra6 = random.randrange(49, 51, self.step)
            pra5 = random.randrange(52, 54, self.step)
            pra2 = random.randrange(48, 50, self.step)
            if pra10 + pra9 + pra6 + pra5 + pra2 <= 370:
                return [pra1, pra2, pra3, pra4, pra5, pra6, pra7, pra8, pra9, pra10]
            else:
                index = True

    def get_random_sample(self):
        pra = self.get_random_pra()
        pr = PatternRects(pra)
        # pr.draw_pattern()
        data = pr.get_details(op_show=False)
        return data

    @staticmethod
    def get_sample(pra, op_show=True, draw_pattern=False):
        pr = PatternRects(pra)
        # pr.draw_pattern()
        if draw_pattern:
            pr.draw_pattern()
        data = pr.get_details(op_show=op_show)
        return data


# 测试生成样本类是否正常工作
def test_get_samples():
    while True:
        gs = GetSamples()
        data = gs.get_random_sample()
        print(data)



if __name__ == '__main__':
    stru = np.ones((400, 400), dtype=int)
    model = FDTDModel(stru)
    data = model.get_results()
    print(data[1])