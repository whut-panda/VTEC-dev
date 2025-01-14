import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from draw_utils import draw_points,draw_xy
from scipy import signal
from collections import Counter

#########################################################
# 原格式转为frames, ids, centers_x, centers_y, widths, heights
def format_convert(data):
    frames = data[:, 0]
    ids = data[:, 1]
    centers_x = data[:, 2] + data[:, 4] / 2
    centers_y = data[:, 3] + data[:, 5] / 2
    widths = data[:, 4]
    heights = data[:, 5]
    convert_data = np.column_stack((frames, ids, centers_x, centers_y, widths, heights,data[:,6:]))
    return convert_data
#########################################################
def trajectory_merge(data):
    traj_temp=np.empty((0, 13))
    for i in range(int(data[:, 1].max())):
        per_id_data = data[data[:, 1] == (i + 1), :]
        per_id_data = per_id_data[np.argsort(per_id_data[:, 0]), :]
        if per_id_data.shape[0]<=10:
            continue

        # 记录轨迹开始时间、坐标和结束时间、坐标
        id = per_id_data[0, 1]
        start_frame, start_x, start_y = per_id_data[0, 0], per_id_data[0, 2], per_id_data[0, 3]
        last_frame, last_x, last_y = per_id_data[-1, 0], per_id_data[-1, 2], per_id_data[-1, 3]
        if start_x < last_x:
            direction = 1
        else:
            direction = -1
        # 记录轨迹的最大坐标和最小坐标、计算每个连续轨迹的长度
        min_x, min_y, max_x, max_y = np.min(per_id_data[:, 2]), np.min(per_id_data[:, 3]), np.max(per_id_data[:, 2]), np.max(per_id_data[:, 3])
        # length=
        per_traj_temp = np.column_stack((id, start_frame, last_frame, last_frame - start_frame,  # 0-3
                                         start_x, start_y, last_x, last_y,  # 4-7,
                                         min_x, min_y, max_x, max_y,  # 8-11
                                         direction,))  # length
        traj_temp = np.row_stack((traj_temp, per_traj_temp))
    traj_parameters = traj_temp[np.argsort(traj_temp[:, 3])[::-1], :]
    print(traj_parameters.shape)

    new_traj_parameter=np.empty((0,traj_parameters.shape[1]))
    last_time =traj_parameters[:,2].max()
    for i in range(int(traj_parameters[:, 0].max())):
        per_traj_parameter=traj_parameters[traj_parameters[:, 0] == (i+1), :]
        if per_traj_parameter.shape[0]==0:
            continue
        if (per_traj_parameter[:,4]<=300 and per_traj_parameter[:,6]>=3700):
            continue
        if per_traj_parameter[:, 4] >=3700 and per_traj_parameter[:, 6] <=300 :
            continue
        if per_traj_parameter[:,1]<=60 and per_traj_parameter[:,6]>=3700:
            continue
        if per_traj_parameter[:,1]<=60 and per_traj_parameter[:,6]>=300:
            continue
        if per_traj_parameter[:, 2] >= (last_time - 60) and per_traj_parameter[:,4]>=3700:
            continue
        if per_traj_parameter[:, 2] >= (last_time - 60) and per_traj_parameter[:,4]<=300:
            continue

        new_traj_parameter=np.row_stack((new_traj_parameter, per_traj_parameter))

    for i in range(int(new_traj_parameter[:, 3].max()),-1,-1):
        #倒序不用i+1
        per_traj_parameter=new_traj_parameter[new_traj_parameter[:, 3] == (i), :]
        if per_traj_parameter.shape[0]==0:
            continue
        per_id_traj_data=data[data[:, 1] == per_traj_parameter[0,0], :]

        #先限定可能的轨迹范围：
        rest_traj_parameters=traj_parameters
        if per_traj_parameter[0,12]==1:
            rest_traj_parameters=rest_traj_parameters[rest_traj_parameters[:, 12]==1, :]
        elif per_traj_parameter[0,12]==-1:
            rest_traj_parameters=rest_traj_parameters[rest_traj_parameters[:, 12]==-1, :]
        #当前轨迹前搜索
        # 空间上筛选，包括
        #不同视频单个车道宽度还不一样，exp20为50够了
        rest1_traj_parameters = rest_traj_parameters[rest_traj_parameters[:, 10] <= per_traj_parameter[0, 8], :]
        rest1_traj_parameters = rest1_traj_parameters[rest1_traj_parameters[:, 9] <= (per_traj_parameter[0, 11]+60),:]
        rest1_traj_parameters = rest1_traj_parameters[rest1_traj_parameters[:, 9] >= (per_traj_parameter[0, 11]-60),:]

        #时间上筛选，包括小于起始帧但大于起始帧少800,800？
        rest1_traj_parameters = rest1_traj_parameters[rest1_traj_parameters[:, 2] <= per_traj_parameter[0,1], :]
        rest1_traj_parameters = rest1_traj_parameters[rest1_traj_parameters[:, 2] >= (per_traj_parameter[0, 1]-800), :]

        # print(rest1_traj_parameters.shape)
        if rest1_traj_parameters.shape[0]==0:
            continue
        else:
            rest1_traj_parameters


        #当前轨迹后搜索
        # 空间上筛选，
        rest2_traj_parameters = rest_traj_parameters[rest_traj_parameters[:, 8] >= per_traj_parameter[0, 10], :]
        rest2_traj_parameters = rest2_traj_parameters[rest2_traj_parameters[:, 9] <= (per_traj_parameter[0, 11] + 60),:]
        rest2_traj_parameters = rest2_traj_parameters[rest2_traj_parameters[:, 11] >= (per_traj_parameter[0, 9] - 60),:]
        #时间上筛选，大于结束帧但小于结束帧多800
        rest2_traj_parameters = rest2_traj_parameters[rest2_traj_parameters[:, 1] >= per_traj_parameter[0, 2], :]
        rest2_traj_parameters = rest2_traj_parameters[rest2_traj_parameters[:, 1] >= (per_traj_parameter[0, 2]+800), :]
        # print(rest2_traj_parameters.shape)


    print(rest2_traj_parameters)
    final_data = np.empty((0, 10))
    for i in range(int(data[:, 1].max())):
        per_id_data = data[data[:, 1] == (i + 1), :]
        if per_id_data.shape[0]<=10:
            continue
        # print(per_id_data[0,0])
        # print(new_traj_parameter[:,0])
        print(np.where(new_traj_parameter[:,0]==per_id_data[0,1])[0].shape)
        print(np.where(new_traj_parameter[:,:]==per_id_data[0,1])[0].shape)
        if np.where(new_traj_parameter[:,0]==per_id_data[0,1])[0].shape[0]==0:
            final_data = np.row_stack((final_data, per_id_data))

    return final_data

#################################################
def repair_coordinates_in_img_edge(per_id_data):
    middle_mean_data=per_id_data
    middle_mean_data=middle_mean_data[middle_mean_data[:,2]>=400,:]
    middle_mean_data=middle_mean_data[middle_mean_data[:,2]<=(3840-400),:]
    middle_mean_data=middle_mean_data[middle_mean_data[:,3]>=100,:]
    middle_mean_data=middle_mean_data[middle_mean_data[:,3]<=(2160-100),:]
    mean_w=np.mean(middle_mean_data[:,4],axis=0)
    mean_h=np.mean(middle_mean_data[:,5],axis=0)
    # for i in range(len(per_id_data[:,0])):
    #     if per_id_data[i, 2] < (mean_w / 2):
    #         per_id_data[i, 2] = per_id_data[i, 2]-((mean_w - per_id_data[i, 4])/2)
    #         per_id_data[i,4]=mean_w
    #     elif per_id_data[i, 2] > (3840-mean_w / 2):
    #         per_id_data[i, 2] = per_id_data[i, 2] + ((mean_w - per_id_data[i, 4]) / 2)
    #         per_id_data[i, 4] = mean_w
    #     if per_id_data[i, 3] < (mean_h / 2):
    #         per_id_data[i, 3] = per_id_data[i, 3]-((mean_h - per_id_data[i, 5])/2)
    #         per_id_data[i,5]=mean_h
    #     elif per_id_data[i, 3] > (2160-mean_h / 2):
    #         per_id_data[i, 3] = per_id_data[i, 3] + ((mean_h - per_id_data[i, 5]) / 2)
    #         per_id_data[i, 5] = mean_h

    per_id_data=per_id_data[per_id_data[:,2]>=(mean_w)]
    per_id_data=per_id_data[per_id_data[:,2]<=(3840-mean_w)]
    per_id_data=per_id_data[per_id_data[:,3]>=(mean_h)]
    per_id_data=per_id_data[per_id_data[:,3]<=(2160-mean_h)]

    return per_id_data

def trajectory_compasation(per_id_data):

    return per_id_data

#############################################
def smoothing_xy_by_SG(x,y,smoothing_window):
    smoothed_x = signal.savgol_filter(x, smoothing_window, 2)
    smoothed_y = signal.savgol_filter(y, smoothing_window, 2)
    return smoothed_x,smoothed_y

if __name__=='__main__':
    filepath = os.path.join('./outputs/', os.listdir('./outputs/')[0])
    # filepath ='./outputs/'+'output.txt'
    print(filepath)

    data = np.loadtxt(filepath,delimiter=',',dtype=bytes).astype(str)
    data = data.astype(np.float64)
    new_data = format_convert(data)  ### 格式从[x1,y1,w,h]转换为[centers_x,centers_y,h,w]
    filename=os.path.basename(filepath)
    # print(new_data.shape)

#############################################################################
    # #### 用于检验，会有同一帧出现两次同一个ID的情况。
    # new_final_data=np.empty((0, 11))
    # for i in range(int(new_data[:, 0].max())):
    #     per_frame_data = new_data[new_data[:, 0] == (i + 1), :]
    #     ids=per_frame_data[:,1]
    #     if ids.duplicated().sum()>=1:
    #         new_per_frame_data = np.empty((0, 11))
    #         for i in ids:
    #             duplicated_id = ids[ids.duplicated() == True]
    #             if i in duplicated_id:
    #                 duplicated_data=per_frame_data[per_frame_data[:,1]==i,:]
    #                 temp_data=duplicated_data[0,:]
    #                 new_per_frame_data = np.row_stack((new_per_frame_data,))
    #             else:
    #                 temp_data=per_frame_data[per_frame_data[:,1]==i,:]
    #                 new_per_frame_data = np.row_stack((new_per_frame_data,temp_data))
    #         new_final_data=np.row_stack((new_final_data, new_per_frame_data))
    #     else:
    #         new_final_data = np.row_stack((new_final_data, per_frame_data))
    # new_data=new_final_data

#############################################################################
    # new_data = trajectory_merge_for_expressway(new_data)
    final_Data = np.empty((0, 9))
    for i in range(int(new_data[:, 1].max())):
        per_id_data = new_data[new_data[:, 1] == (i + 1), :]
        per_id_data = per_id_data[np.argsort(per_id_data[:, 0]), :]
        if per_id_data.shape[0]<=10:
            continue
        min_x, min_y, max_x, max_y = np.min(per_id_data[:, 2]), np.min(per_id_data[:, 3]), np.max(per_id_data[:, 2]), np.max(per_id_data[:, 3])
        if max_x-min_x <=100:
            continue
        if per_id_data.shape[0]<=10:
            continue

        # per_id_data=repair_coordinates_in_img_edge(per_id_data)
        # per_id_data=trajectory_compasation(per_id_data)

        # if per_id_data.shape[0]<=32:
        #     continue

        smoothed_x, smoothed_y = smoothing_xy_by_SG(per_id_data[:, 2], per_id_data[:, 3], smoothing_window=31)
        # smoothed_x, smoothed_y = (per_id_data[:, 2], per_id_data[:, 3])

        vehicle_class=0
        # vehicle_class= Counter(per_id_data[:,6]).most_common(1)[0][0]
        vehicle_class_column=np.full(per_id_data[:,6].shape, vehicle_class)
        new_per_id_data=np.column_stack((per_id_data[:,:6], smoothed_x, smoothed_y,vehicle_class_column))
        # print(new_per_id_data.shape)
        final_Data = np.row_stack((final_Data, new_per_id_data))

    # draw_xy(new_data,2,3)
    # print(sortData)
    save_path='./reconstruction_txts/' + filename[:7] + '_reconstructed.txt'
    np.savetxt(save_path, final_Data,
               fmt=['%0.0f', '%0.0f', '%0.4f', '%0.4f', '%0.4f', '%0.4f', # frame, id, centers_x, centers_y, width, height, 0-5
                    '%0.4f', '%0.4f', '%0.0f'# denoised_x, denoised_y, 6-7, class, 8,共9个数据
                    ],delimiter=',')