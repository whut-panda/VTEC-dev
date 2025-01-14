import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn import metrics
from sklearn.cluster import DBSCAN
import math
from PIL import Image
import os
from scipy import signal

def getline_k_b(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    k,b=-a/b,-c/b
    return [k,b]

# 根据已知两点坐标，求过这两点的直线解析方程： a*x+b*y+c = 0  (a >= 0)
def getLinearEquation(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]

# 根据直线的起点与终点计算出平行距离D的平行线的方程
def getLinearEquation(p1x, p1y, p2x, p2y, distance):
    """
    :param p1x: 起点X
    :param p1y: 起点Y
    :param p2x: 终点X
    :param p2y: 终点Y
    :param distance: 平距
    :param left_right: 向左还是向右
    """
    e = getLinearEquation(p1x, p1y, p2x, p2y)
    f = distance * math.sqrt(e.a * e.a + e.b * e.b)
    m1 = e.c + f
    m2 = e.c - f
    # result = 值1 if 条件 else 值2
    c2 = m1 if p2y - p1y < 0 else m2
    return [e.a, e.b, c2]

#####通过车道线标定点
####################################################################
class LaneDetectionByCoordinates:
    def __init__(self,Data):
        self.Data=Data

    def for_expressway0104(self):
        data = self.Data
        new_data = np.column_stack((data, np.zeros((data.shape[0], 1))))
        up3 = np.poly1d(getline_k_b(97, 991, 6418, 953))
        up2 = np.poly1d(getline_k_b(97, 1030, 6418, 993))
        up1 = np.poly1d(getline_k_b(97, 1070, 6418, 1034))
        middle = np.poly1d(getline_k_b(97, 1119, 6418, 1085))
        down1 = np.poly1d(getline_k_b(97, 1171, 6418, 1130))
        down2 = np.poly1d(getline_k_b(97, 1210, 6418, 1177))
        down3 = np.poly1d(getline_k_b(97, 1250, 6418, 1217))

        # 从下网上车道号1-6
        for i in range(new_data.shape[0]):
            middle_y = middle(new_data[i, 6])
            up1_y = up1(new_data[i, 6])
            up2_y = up2(new_data[i, 6])
            up3_y = up3(new_data[i, 6])
            down1_y = down1(new_data[i, 6])
            down2_y = down2(new_data[i, 6])
            down3_y = down3(new_data[i, 6])
            temp = new_data[i, 7]
            if temp <= middle_y:
                if temp > up1_y:
                    new_data[i, 9] = 4
                elif temp <= up1_y and temp > up2_y:
                    new_data[i, 9] = 3
                elif temp <= up2_y and temp > up3_y:
                    new_data[i, 9] = 2
                else:
                    new_data[i, 9] = 1
            else:
                if temp <= down1_y:
                    new_data[i, 9] = 5
                elif temp > down1_y and temp <= down2_y:
                    new_data[i, 9] = 6
                elif temp > down2_y and temp <= down3_y:
                    new_data[i, 9] = 7
                else:
                    new_data[i, 9] = 8
        return new_data

    def for_expressway0106(self):
        data = self.Data
        new_data = np.column_stack((data, np.zeros((data.shape[0], 1))))
        up3 = np.poly1d(getline_k_b(129, 961, 6544, 939))
        up2 = np.poly1d(getline_k_b(129, 1002, 6544, 982))
        up1 = np.poly1d(getline_k_b(129, 1044, 6544, 1025))
        middle = np.poly1d(getline_k_b(129, 1096, 6544, 1078))
        down1 = np.poly1d(getline_k_b(186, 1149, 6544, 1132))
        down2 = np.poly1d(getline_k_b(186, 1190, 6544, 1174))
        down3 = np.poly1d(getline_k_b(186, 1232, 6544, 1217))

        # 从下网上车道号1-6
        for i in range(new_data.shape[0]):
            middle_y = middle(new_data[i, 6])
            up1_y = up1(new_data[i, 6])
            up2_y = up2(new_data[i, 6])
            up3_y = up3(new_data[i, 6])
            down1_y = down1(new_data[i, 6])
            down2_y = down2(new_data[i, 6])
            down3_y = down3(new_data[i, 6])
            temp = new_data[i, 7]
            if temp <= middle_y:
                if temp > up1_y:
                    new_data[i, 9] = 4
                elif temp <= up1_y and temp > up2_y:
                    new_data[i, 9] = 3
                elif temp <= up2_y and temp > up3_y:
                    new_data[i, 9] = 2
                else:
                    new_data[i, 9] = 1
            else:
                if temp <= down1_y:
                    new_data[i, 9] = 5
                elif temp > down1_y and temp <= down2_y:
                    new_data[i, 9] = 6
                elif temp > down2_y and temp <= down3_y:
                    new_data[i, 9] = 7
                else:
                    new_data[i, 9] = 8
        return new_data

    def for_curve0201(self):
        new_data=self.Data
        new_data = np.column_stack((new_data, np.zeros((new_data.shape[0], 1))))
        x_array1, y_array1 = [30, 930, 1585, 2042, 2710, 3392, 3691], [1118, 1103, 1069, 1030, 942, 830, 772]
        x_array2, y_array2 = [26, 972, 1668, 2794, 3702], [1172, 1152, 1113, 981, 822]
        x_array3, y_array3 = [13, 1016, 1458, 2378, 3764], [1243, 1224, 1200, 1113, 888]
        x_array4, y_array4 = [60, 1172, 1649, 2326, 3000, 3812], [1311, 1287, 1257, 1189, 1093, 950]
        x_array5, y_array5 = [84, 746, 1439, 2107, 3008, 3823], [1363, 1353, 1325, 1268, 1143, 1002]
        x_array6, y_array6 = [19, 671, 1348, 2275, 2803, 3772], [1420, 1356, 1387, 1305, 1234, 1066]
        f_x1 = np.polyfit(x_array1, y_array1, 3)
        f_x2 = np.polyfit(x_array2, y_array2, 3)
        f_x3 = np.polyfit(x_array3, y_array3, 3)
        f_x4 = np.polyfit(x_array4, y_array4, 3)
        f_x5 = np.polyfit(x_array5, y_array5, 3)
        f_x6 = np.polyfit(x_array6, y_array6, 3)
        p_x1 = np.poly1d(f_x1)
        p_x2 = np.poly1d(f_x2)
        p_x3 = np.poly1d(f_x3)
        p_x4 = np.poly1d(f_x4)
        p_x5 = np.poly1d(f_x5)
        p_x6 = np.poly1d(f_x6)
        curve_y1 = p_x1(new_data[:, 6])
        curve_y2 = p_x2(new_data[:, 6])
        curve_y3 = p_x3(new_data[:, 6])
        curve_y4 = p_x4(new_data[:, 6])
        curve_y5 = p_x5(new_data[:, 6])
        curve_y6 = p_x6(new_data[:, 6])
        for i in range(new_data.shape[0]):
            temp = new_data[i, 7]
            if temp <= curve_y3[i]:
                if temp > curve_y2[i]:
                    new_data[i, 9] = 3
                elif temp <= curve_y2[i] and temp > curve_y1[i]:
                    new_data[i, 9] = 2
                else:
                    new_data[i, 9] = 1
            else:
                if temp <= curve_y4[i]:
                    new_data[i, 9] = 4
                elif temp > curve_y4[i] and temp <= curve_y5[i]:
                    new_data[i, 9] = 5
                elif temp > curve_y5[i] and temp <= curve_y6[i]:
                    new_data[i, 9] = 6
                else:
                    new_data[i, 9] = 7
        return new_data

    def for_merge0301(self):
        data = self.Data
        new_data = np.column_stack((data, np.zeros((data.shape[0], 1))))
        up3 = np.poly1d(getline_k_b(153, 712, 3840, 721))
        up2 = np.poly1d(getline_k_b(63, 769, 3840, 777))
        up1 = np.poly1d(getline_k_b(72, 821, 3840, 830))
        middle = np.poly1d(getline_k_b(114, 891, 3840, 899))
        down1 = np.poly1d(getline_k_b(33, 962, 3840, 971))
        down2 = np.poly1d(getline_k_b(41, 1014, 3840, 1023))
        down3 = np.poly1d(getline_k_b(31, 1070, 3840, 1081))

        # 从下网上车道号1-8
        for i in range(new_data.shape[0]):
            middle_y = middle(new_data[i, 6])
            up1_y = up1(new_data[i, 6])
            up2_y = up2(new_data[i, 6])
            up3_y = up3(new_data[i, 6])
            down1_y = down1(new_data[i, 6])
            down2_y = down2(new_data[i, 6])
            down3_y = down3(new_data[i, 6])
            # temp=p_y_x(new_data[i,6])
            temp = new_data[i, 7]
            if temp <= middle_y:
                if temp > up1_y:
                    new_data[i, 9] = 4
                elif temp <= up1_y and temp > up2_y:
                    new_data[i, 9] = 3
                elif temp <= up2_y and temp > up3_y:
                    new_data[i, 9] = 2
                else:
                    new_data[i, 9] = 1
            else:
                if temp <= down1_y:
                    new_data[i, 9] = 5
                elif temp > down1_y and temp <= down2_y:
                    new_data[i, 9] = 6
                elif temp > down2_y and temp <= down3_y:
                    new_data[i, 9] = 7
                else:
                    new_data[i, 9] = 8
        return new_data

    def for_merge0304(self):
        new_data=self.Data
        new_data = np.column_stack((new_data, np.zeros((new_data.shape[0], 1))))
        up3 = np.poly1d(getline_k_b(124, 888, 3690, 891))
        up2 = np.poly1d(getline_k_b(151, 972, 3732, 972))
        up1 = np.poly1d(getline_k_b(156, 1050, 3734, 1047))
        middle = np.poly1d(getline_k_b(199, 1157, 3791, 1147))
        down1 = np.poly1d(getline_k_b(167, 1257, 3732, 1248))
        down2 = np.poly1d(getline_k_b(163, 1335, 3734, 1323))
        down3 = np.poly1d(getline_k_b(140, 1416, 3793, 1403))

        # 从下往上车道号1-8
        for i in range(new_data.shape[0]):
            middle_y = middle(new_data[i, 6])
            up1_y = up1(new_data[i, 6])
            up2_y = up2(new_data[i, 6])
            up3_y = up3(new_data[i, 6])
            down1_y = down1(new_data[i, 6])
            down2_y = down2(new_data[i, 6])
            down3_y = down3(new_data[i, 6])
            # temp=p_y_x(new_data[i,6])
            temp = new_data[i, 7]
            if temp <= middle_y:
                if temp > up1_y:
                    new_data[i, 9] = 4
                elif temp <= up1_y and temp > up2_y:
                    new_data[i, 9] = 3
                elif temp <= up2_y and temp > up3_y:
                    new_data[i, 9] = 2
                else:
                    new_data[i, 9] = 1
            else:
                if temp <= down1_y:
                    new_data[i, 9] = 5
                elif temp > down1_y and temp <= down2_y:
                    new_data[i, 9] = 6
                elif temp > down2_y and temp <= down3_y:
                    new_data[i, 9] = 7
                else:
                    new_data[i, 9] = 8
        return new_data
    def for_long0501(self):
        new_data=self.Data
        new_data = np.column_stack((new_data, np.zeros((new_data.shape[0], 1))))
        x_array1, y_array1 = [1649,2052,2472,3051,3841,4488,5044,5423,5623,5762,5962,6158,6306,6509,6721,6877,7033,], \
                             [603,599,599,601,599,595,597,592,590,590,590,588,586,586,584,582,580]
        x_array2, y_array2 = [64, 131, 232, 296, 396, 461, 556,720,787,884,949,1048,1216,1381,1715, 1883,1956,2044,2208,2448,2603,2763,2903,3063,3230,3394,3566,3722,3877,4045,4201,4393,4564,4708,4884,5052,5223,5383,5539,5719,5874,6042,6214,6370,6553,6721,6905], \
                             [779, 770, 756, 748, 736, 729, 718,701,696,688,685,678,669, 662, 676, 654,652,652,650,650,652,650,650,652,654,654,652,650,648,650,645,645,645,643,639,637,635,633,633,631,631,629,629,627,627,625,623]
        x_array3, y_array3 = [39,135,239,331,443,555,658,758,854,954,1038,1118,1230,1337,1449,1565,1729,1893,2064,2236,2404,2571,2731,2899,3059,3214,3374,3538,3706,3857,4025,4185,4365,4524,4692,4860,5020,5195,5351,5511,5675,5846,6018,6194,6354,6533,6705,6857,], \
                             [823, 806,794,780,768,756,743,735,727,721,715,711,705,703,698,694,694,688,690,690,688,688,690,690,690,690,690,690,690,688,686,684,684,682,680,680,678,676,674,674,674,672,672,672,670,668,666,666]
        x_array4, y_array4 = [19,115,327,635,1022,1385,1853,2384,2851,3302,3849,4373,4952,5427,5818,6290,6649,6929,], \
                             [874,857,827,796,764,751,743,741,737,739,737,733,727,725,725,723,719,713]
        # x_array5, y_array5 = [84, 746, 1439, 2107, 3008, 3823], [1363, 1353, 1325, 1268, 1143, 1002]
        # x_array6, y_array6 = [19, 671, 1348, 2275, 2803, 3772], [1420, 1356, 1387, 1305, 1234, 1066]
        f_x1 = np.polyfit(x_array1, y_array1, 8)
        f_x2 = np.polyfit(x_array2, y_array2, 8)
        f_x3 = np.polyfit(x_array3, y_array3, 8)
        f_x4 = np.polyfit(x_array4, y_array4, 8)
        # f_x5 = np.polyfit(x_array5, y_array5, 3)
        # f_x6 = np.polyfit(x_array6, y_array6, 3)
        p_x1 = np.poly1d(f_x1)
        p_x2 = np.poly1d(f_x2)
        p_x3 = np.poly1d(f_x3)
        p_x4 = np.poly1d(f_x4)
        # p_x5 = np.poly1d(f_x5)
        # p_x6 = np.poly1d(f_x6)
        curve_y1 = p_x1(new_data[:, 6])
        curve_y2 = p_x2(new_data[:, 6])
        curve_y3 = p_x3(new_data[:, 6])
        curve_y4 = p_x4(new_data[:, 6])
        # curve_y5 = p_x5(new_data[:, 6])
        # curve_y6 = p_x6(new_data[:, 6])
        for i in range(new_data.shape[0]):
            temp = new_data[i, 7]
            if temp <= curve_y4[i]:
                if temp > curve_y3[i]:
                    new_data[i, 9] = 4
                elif temp <= curve_y3[i] and temp > curve_y2[i]:
                    new_data[i, 9] = 3
                elif temp <= curve_y2[i] and temp > curve_y1[i]:
                    new_data[i, 9] = 2
                else:
                    new_data[i, 9] = 1
            # else:
                # if temp <= curve_y4[i]:
                #     new_data[i, 9] = 4
                # elif temp > curve_y4[i] and temp <= curve_y5[i]:
                #     new_data[i, 9] = 5
                # elif temp > curve_y5[i] and temp <= curve_y6[i]:
                #     new_data[i, 9] = 6
                # else:
                #     new_data[i, 9] = 7
        return new_data
############################################################

#####通过自动提取车道线
def lane_detection_for_expressway_by_cluster(data):

    return None
####################
def draw_lane(data,img_path=None):
    plt.figure(figsize=(10, 7))
    plt.title('lane data')
    color=['blue','brown','green','pink','orange','red','yellow','purple',]
    n=int(data[:, 9].max())
    for i in range(0,n):
        plt.scatter(data[data[:,9]==(i+1),6], data[data[:,9]==(i+1),7], c=color[i], s=1,linewidths=1,label='lane'+' '+str(i+1))
    # plt.plot(x,y0,c='black',label='Road divider')
    plt.ylabel("Y/pixel")
    plt.xlabel("X/pixel")
    plt.legend(loc="upper left")
    if img_path is not None:
        img = Image.open(img_path)
        plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    # 01:expressway, 02:curve, 03:merge, 04:intersection
    txt_path='../../06Tracking-by-Detection/resort_and_reconstruction/reconstruction_txts/exp0607_reconstructed.txt'
    data = np.loadtxt(txt_path,delimiter=',',dtype=bytes).astype(str)
    data = data.astype(np.float64)
    filename=os.path.basename(txt_path)

    # if int(filename[4:7]) == 104:
    #     new_data = LaneDetectionByCoordinates(data).for_expressway0104()
    #     draw_lane(new_data,img_path='./background_img/exp0104-000001--.png')
    # if int(filename[4:7]) == 106:
    #     new_data = LaneDetectionByCoordinates(data).for_expressway0106()
    #     draw_lane(new_data,img_path='./background_img/exp0106_0005--.png')
    # if int(filename[4:7]) == 108:
    #     new_data = LaneDetectionByCoordinates(data).for_expressway0106()
    #     draw_lane(new_data,img_path='./background_img/exp0108_0005--.png')
    # if int(filename[4:7]) == 201:
    #     new_data = LaneDetectionByCoordinates(data).for_curve0201()
    #     draw_lane(new_data,img_path='./background_img/exp0201-000247--.png')
    # if int(filename[4:7]) == 301:
    #     new_data = LaneDetectionByCoordinates(data).for_merge0301()
    #     draw_lane(new_data,img_path='./background_img/exp0301-000005--.png')
    # if int(filename[4:7]) == 305:
    #     new_data = LaneDetectionByCoordinates(data).for_merge0304()
    #     draw_lane(new_data,img_path='./background_img/exp0305-000101--.png')
    # if int(filename[4:7]) == 502:
    #     new_data = LaneDetectionByCoordinates(data).for_expressway0104()
    #     draw_lane(new_data,img_path='./background_img/stitched_image0502.jpg')
    # if int(filename[4:7]) == 503:
    #     new_data = LaneDetectionByCoordinates(data).for_expressway0106()
    #     draw_lane(new_data,img_path='./background_img/stitched_image0503.jpg')
    # if int(filename[4:6]) == 60:
    #     new_data = np.column_stack((data, np.zeros((data.shape[0], 1))))
    #     new_data[:,9]=2

    new_data = np.column_stack((data, np.zeros((data.shape[0], 1))))
    new_data[:,9]=2


    save_path = './01lane_detection_txts/' + filename[:7] + '_lane_detection.txt'
    np.savetxt(save_path, new_data,
               fmt=['%0.0f', '%0.0f', '%0.4f', '%0.4f', '%0.4f', '%0.4f', # frame, id, centers_x, centers_y, width, height, 0-5
                    '%0.4f', '%0.4f', '%0.0f', '%0.0f', # denoised_x, denoised_y, 6-7, class,8, lane, 9 共10个
                    ],delimiter=',')
