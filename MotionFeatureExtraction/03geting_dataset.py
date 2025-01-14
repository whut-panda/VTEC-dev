import os
import numpy as np
from draw_utils import draw_xy
import pandas as pd

# 单位转换：坐标、时间
def unit_conversion(data,ratio):
    transformed_time = data[:, 0]/ 30     # 时间转化，帧率30
    data0_2=data[:,0:2]
    data6_8=data[:,6:8]*ratio
    data4_6=data[:,4:6]*ratio

    data8_10=data[:,8:10]
    data10_13=data[:,10:13]*ratio* 30 #速度
    data13_16=data[:,13:16]*ratio*30*30 #加速度
    data16_18=data[:,16:18]
    data18=data[:,18]*ratio
    data19_21=data[:,19:21]/30
    transformed_data = np.column_stack((data0_2,data[:,6:8],data[:,4:6],data6_8,data4_6,
                                        data8_10,data10_13,data13_16,data16_18,data18,data19_21,
                                        transformed_time))
    return transformed_data

def data_filter(data,traj_thredshold=250,interval1=0,interval2=-1):
    a, b =data.shape
    new_data = np.empty((0,b))
    for i in range(int(data[:, 1].max())):
        per_id_data = data[data[:, 1] == (i + 1), :]
        per_id_data = per_id_data[np.argsort(per_id_data[:, 0]), :]
        if per_id_data.shape[0]<=traj_thredshold:
            continue
        new_per_id_data = per_id_data[interval1:interval2, :]
        new_data = np.row_stack((new_data, new_per_id_data))
    return new_data


if __name__=='__main__':
    # 01:expressway, 02:curve, 03:merge, 04:intersection
    txt_path='./02va_headway_extraction_txts/exp0607_va_headway_extraction.txt'
    data = np.loadtxt(txt_path,delimiter=',',dtype=bytes).astype(str)
    data = data.astype(np.float64)
    filename=os.path.basename(txt_path)

    if int(filename[4:7]) == 108:
        ratio = 285/3514
    if int(filename[4:7]) == 109:
        ratio = 270/3563
    if int(filename[4:7]) == 101:
        ratio = 360/3499
    if int(filename[4:7]) == 102:
        ratio = 255/3512
    if int(filename[4:7]) == 302:
        ratio = 225 / 3344
    if int(filename[4:7]) == 303:
        ratio = 240/ 3620
    if int(filename[4:7]) == 304:
        ratio = 225/3528
    if int(filename[4:7]) == 105:
        ratio = 210/3246
    if int(filename[4:7]) == 103:
        ratio = 300/3556
    if int(filename[4:7]) == 107:
        ratio = 405/3619
    if int(filename[4:7]) == 501:
        ratio = 240/2601
    if int(filename[4:7]) == 502:
        ratio = 555/6311
    if int(filename[4:7]) == 503:
        ratio = 570/6733
    if int(filename[4:6]) == 60:
        # ratio = 36/411 #601
        # ratio = 132/1542 #602
        # ratio = 54/613 #603
        # ratio = 84/986 #604
        # ratio = 180/2113 #605
        # ratio = 204/2385 #606
        ratio = 42/490 #607

    # data = data[data[:,6]>=100,:]
    # data = data[data[:,6]<=3740,:]
    # data1= data[data[:,24]==1,:]
    # data2= data[data[:,24]==2,:]
    # data3= data[data[:,24]==3,:]
    # data4= data[data[:,24]==4,:]
    # data5= data[data[:,24]==5,:]
    # data6= data[data[:,24]==6,:]
    # data =np.column_stack((data1,data2,data3,data4,data5,data6))


    data = data_filter(data, traj_thredshold=150, interval1=0, interval2=-1)
    new_data = unit_conversion(data, ratio)
    print(new_data.shape)

    # new_data=new_data[new_data[:,24]!=7,:]

    save_path1 = './03dataset_txts/' + filename[:7] + '_dataset.txt'
    save_path2 = './03dataset_txts/' + filename[:7] + '_dataset.csv'
    np.savetxt(save_path1, new_data,
               fmt=['%0.0f', '%0.0f', '%0.4f', '%0.4f', '%0.4f', '%0.4f',
                    '%0.4f', '%0.4f', '%0.4f', '%0.4f', '%0.0f', '%0.0f',
                    '%0.4f', '%0.4f', '%0.4f', '%0.4f', '%0.4f', '%0.4f',
                    '%0.0f', '%0.0f', '%0.4f', '%0.4f', '%0.4f', '%0.4f'],delimiter=',')
    columns =['frame', 'id', 'center_x_pixel', 'center_y_pixel', 'width_pixel', 'height_pixel', #0-5
              'center_x', 'center_y', 'width', 'height', 'class','lane',                  #6-9,10,11
              'calculated_vx','calculated_vy','calculated_v', 'calculated_ax','calculated_ay','calculated_a', #12-14,15-17
              'preceding_id','following_id','DHW','THW','TTC','transformed_time'] #18,19,20-22,23,总共有24个
    data2 = pd.read_csv(save_path1, delimiter=",",header=None,index_col=False,names=columns)
    data2.to_csv(save_path2, encoding='utf-8', index=False)
