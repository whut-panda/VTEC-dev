import os
import numpy as np
from draw_utils import draw_xy
from multiprocessing import Pool,cpu_count,Process
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
import numexpr


def calculating_v(time_values,smoothed_x,smoothed_y):
    # 添加参数
    initial_vel = 0.

    # create matrix A containing current time values, and matrix B containing next time values
    t_matrix = time_values
    t_matrix_B = t_matrix[1:]
    t_matrix_A = t_matrix[0:-1]
    # create matrix of A containing current x and y and matrix B containing next x and y values
    x_y_matrix = np.column_stack((smoothed_x, smoothed_y))
    x_y_matrix_B = x_y_matrix[1:, :]
    x_y_matrix_A = x_y_matrix[0:-1, :]

    delta_t = numexpr.evaluate('(t_matrix_B - t_matrix_A)')
    delta_x_y = numexpr.evaluate('(x_y_matrix_B - x_y_matrix_A)')
    delta_x, delta_y = delta_x_y[:, 0], delta_x_y[:, 1]
    vel_x = numexpr.evaluate('delta_x/ (t_matrix_B - t_matrix_A)')
    vel_y = numexpr.evaluate('delta_y/ (t_matrix_B - t_matrix_A)')

    #合速度计算还得再考虑下
    dist_temp = numexpr.evaluate('sum((x_y_matrix_B - x_y_matrix_A)**2, 1)')
    dist = numexpr.evaluate('sqrt(dist_temp)')
    vel = numexpr.evaluate('dist/ (t_matrix_B - t_matrix_A)')
    # vel = numexpr.evaluate('sqrt(vel_x**2 + vel_y**2)')

    vel_x = np.insert(vel_x, 0, initial_vel, axis=0)
    vel_y = np.insert(vel_y, 0, initial_vel, axis=0)
    vel = np.insert(vel, 0, initial_vel, axis=0)
    calculated_velocities = np.column_stack((vel_x,vel_y,vel))
    return calculated_velocities

def modify_vel_initial_value(calculated_velocities):
    f_v_x0 = np.polyfit([1.0,2.0,3.0,4.0,5.0], calculated_velocities[1:6,0], 1)
    p_v_x0 = np.poly1d(f_v_x0)
    calculated_velocities[0:1,0] = p_v_x0([0])

    f_v_x1 = np.polyfit([1.0,2.0,3.0,4.0,5.0], calculated_velocities[1:6, 1], 1)
    p_v_x1 = np.poly1d(f_v_x1)
    calculated_velocities[0:1, 1] = p_v_x1([0])

    f_v_x2 = np.polyfit([1.0,2.0,3.0,4.0,5.0], calculated_velocities[1:6, 2], 1)
    p_v_x2 = np.poly1d(f_v_x2)
    calculated_velocities[0:1, 2] = p_v_x2([0])
    return calculated_velocities

def smoothing_velocities(calculated_velocities,smoothing_window):
    temp_0 = signal.savgol_filter(calculated_velocities[:, 0], smoothing_window, 3)
    temp_1 = signal.savgol_filter(calculated_velocities[:, 1], smoothing_window, 3)
    temp_2 = signal.savgol_filter(calculated_velocities[:, 2], smoothing_window, 3)
    smoothed_velocities=np.column_stack((temp_0[:,None],temp_1[:,None],temp_2[:,None]))
    return smoothed_velocities

def calculating_a(time_values,velocities):
    initial_accel = 0.

    t_matrix = time_values
    t_matrix_B = t_matrix[1:]
    t_matrix_A = t_matrix[0:-1]

    vel_matrix = velocities
    vel_matrix_B = vel_matrix[1:,:]
    vel_matrix_A = vel_matrix[0:-1,:]

    delta_t = numexpr.evaluate('(t_matrix_B - t_matrix_A)')
    delta_vx_vy_v = numexpr.evaluate('(vel_matrix_B - vel_matrix_A)')
    delta_vx, delta_vy,delta_v = delta_vx_vy_v[:, 0], delta_vx_vy_v[:, 1],delta_vx_vy_v[:,2]
    a_x = numexpr.evaluate('delta_vx/ (t_matrix_B - t_matrix_A)')
    a_y = numexpr.evaluate('delta_vy/ (t_matrix_B - t_matrix_A)')
    #合加速度利用合速度计算
    a = numexpr.evaluate('delta_v/ (t_matrix_B - t_matrix_A)')
    a_x = np.insert(a_x, 0, initial_accel, axis=0)
    a_y = np.insert(a_y, 0, initial_accel, axis=0)
    a = np.insert(a, 0, initial_accel, axis=0)
    calculated_accelerations = np.column_stack((a_x, a_y,a))
    return calculated_accelerations

def modify_accel_initial_value(calculated_accelerations):
    f_v_x0 = np.polyfit([1.0, 2.0, 3.0, 4.0, 5.0], calculated_accelerations[1:6,0], 1)
    p_v_x0 = np.poly1d(f_v_x0)
    calculated_accelerations[0,0] = p_v_x0(0)

    f_v_x1 = np.polyfit([1.0, 2.0, 3.0, 4.0, 5.0], calculated_accelerations[1:6,1], 1)
    p_v_x1 = np.poly1d(f_v_x1)
    calculated_accelerations[0,1] = p_v_x1(0)

    f_v_x2 = np.polyfit([1.0, 2.0, 3.0, 4.0, 5.0], calculated_accelerations[1:6,2], 1)
    p_v_x2 = np.poly1d(f_v_x2)
    calculated_accelerations[0,2] = p_v_x2(0)
    return calculated_accelerations

def smoothing_accelerations(calculated_accelerations,smoothing_window):
    temp_0 = signal.savgol_filter(calculated_accelerations[:, 0], smoothing_window, 3)
    temp_1 = signal.savgol_filter(calculated_accelerations[:, 1], smoothing_window, 3)
    temp_2 = signal.savgol_filter(calculated_accelerations[:, 2], smoothing_window, 3)
    smoothed_accelerations=np.column_stack((temp_0[:,None],temp_1[:,None],temp_2[:,None]))
    return smoothed_accelerations

########################################################################
#基于车道数据提取，因此可用于expressway,curve,merge
def new_per_frame_data_adding_pf_id(per_frame_data):
    new_per_frame_data= np.empty((0, 18))
    for j in range(int(per_frame_data[:, 9].max())):
        per_lane_data = per_frame_data[per_frame_data[:, 9] == (j + 1), :]
        if per_lane_data.shape[0] == 0:
            continue
        if np.mean(per_lane_data[:,10])>=0: #用v_x判断车道流向
            per_lane_data = per_lane_data[np.argsort(per_lane_data[:, 6]), :] #利用车辆位置提取车道车辆排列
        else:
            per_lane_data = per_lane_data[np.argsort(-per_lane_data[:, 6]), :]
        per_lane_id_numbers = per_lane_data.shape[0]
        per_lane_id_array = list(per_lane_data[:, 1])

        for index, k in enumerate(per_lane_id_array):
            per_id_data = per_lane_data[per_lane_data[:, 1] == k, :]
            if per_lane_id_numbers == 1:
                preceding_id = 0
                following_id = 0
            else:
                if index == 0:
                    preceding_id = per_lane_id_array[index + 1]
                    following_id = 0
                elif index == per_lane_id_numbers - 1:
                    preceding_id = 0
                    following_id = per_lane_id_array[index - 1]
                else:
                    preceding_id = per_lane_id_array[index + 1]
                    following_id = per_lane_id_array[index - 1]
            new_per_id_data=np.column_stack((per_id_data, preceding_id,following_id))
            new_per_frame_data = np.row_stack((new_per_frame_data, new_per_id_data))
    return new_per_frame_data

# 也可用于expressway,curve,merge?
def new_per_frame_data_adding_headway(per_frame_data):
    new_per_frame_data= np.empty((0, 21))
    for j in range(int(per_frame_data[:, 1].max())):
        per_id_data = per_frame_data[per_frame_data[:, 1] == (j + 1), :]
        if per_id_data.shape[0] == 0:
            continue
        preceding_id, following_id = int(per_id_data[0, 16]), int(per_id_data[0, 17])
        preceding_id_data = per_frame_data[per_frame_data[:, 1] == preceding_id, :]
        if preceding_id == 0:
            curent_preceding_headway=0
            time_headway=0
            time_to_collision=0
            new_per_id_data = np.column_stack((per_id_data, curent_preceding_headway,time_headway,time_to_collision))
        else:
            ###计算的还都是两车间的垂直数据，弯曲道路还未考虑
            # 车头间距 distance headway
            if per_id_data[0, 6]<preceding_id_data[0,6]:
                curent_head_x=per_id_data[0, 6] + per_id_data[0,4]/2
                curent_head_y=per_id_data[0, 7]  #严格来说，车辆宽度得考虑
                preceding_head_x = preceding_id_data[0,6] + preceding_id_data[0,4]/2
                preceding_head_y = preceding_id_data[0,7]  #严格来说，车辆宽度得考虑
                curent_preceding_headway=np.sqrt(np.square(curent_head_x-preceding_head_x)+
                                                 np.square(curent_head_y-preceding_head_y))
                preceding_trail_x=preceding_head_x - preceding_id_data[0,4]
                preceding_trail_y=preceding_head_y
                curent_preceding_distance=np.sqrt(np.square(curent_head_x-preceding_trail_x)+
                                                 np.square(curent_head_y-preceding_trail_y))
            else:
                curent_head_x=per_id_data[0, 6] - per_id_data[0,4]/2
                curent_head_y=per_id_data[0, 7]
                preceding_head_x = preceding_id_data[0,6] - preceding_id_data[0,4]/2
                preceding_head_y = preceding_id_data[0,7]
                curent_preceding_headway=np.sqrt(np.square(curent_head_x-preceding_head_x)+
                                                 np.square(curent_head_y-preceding_head_y))
                preceding_trail_x=preceding_head_x + preceding_id_data[0,4]
                preceding_trail_y=preceding_head_y
                curent_preceding_distance=np.sqrt(np.square(curent_head_x-preceding_trail_x)+
                                                 np.square(curent_head_y-preceding_trail_y))
            # 车头时距THW, time headway
            current_v = per_id_data[0,12]
            time_headway=curent_preceding_headway /current_v

            # TTC, time to collision
            current_v = per_id_data[0, 12]
            preceding_v=preceding_id_data[0,12]
            delta_v=current_v-preceding_v
            if delta_v<=0:
                time_to_collision=0
            else:
                time_to_collision=curent_preceding_distance/delta_v
            new_per_id_data = np.column_stack((per_id_data, curent_preceding_headway,time_headway,time_to_collision))
        new_per_frame_data = np.row_stack((new_per_frame_data, new_per_id_data))
    return new_per_frame_data

if __name__=='__main__':
    # 01:expressway, 02:curve, 03:merge, 04:intersection
    txt_path='./01lane_detection_txts/exp0607_lane_detection.txt'
    data = np.loadtxt(txt_path, delimiter=',',dtype=bytes).astype(str)
    new_data = data.astype(np.float64)
    new_data = new_data[new_data[:,0]<=10000,:]
    filename=os.path.basename(txt_path)

################################################
    time5 = time.time()
    final_Data = np.empty((0, 16))
    for i in range(int(new_data[:, 1].max())):
        per_id_data = new_data[new_data[:, 1] == (i + 1), :]
        per_id_data = per_id_data[np.argsort(per_id_data[:, 0]), :]
        if per_id_data.shape[0]<=10:
            continue
        time_values=per_id_data[:,0]
        smoothed_x, smoothed_y = per_id_data[:, 6], per_id_data[:, 7]

        ### 基于平滑数据计算v，a
        calculated_velocities=calculating_v(time_values,smoothed_x,smoothed_y)
        calculated_velocities=modify_vel_initial_value(calculated_velocities)
        # smoothed_velocities = smoothing_velocities(calculated_velocities,smoothing_window=41)

        # 根据速度计算加速度
        calculated_accelerations = calculating_a(time_values,calculated_velocities)
        calculated_accelerations=modify_accel_initial_value(calculated_accelerations)
        # smoothed_accelarations = smoothing_accelerations(calculated_accelerations,smoothing_window=41)

        new_per_id_data=np.column_stack((per_id_data, calculated_velocities, calculated_accelerations))
        final_Data = np.row_stack((final_Data, new_per_id_data))
    time6 = time.time()
    print(str(time6 - time5) + 's')
    print(final_Data.shape)
    print('-')

##################################################
    new_data=final_Data
    time1 = time.time()
    task_list1=list()
    for i in range(int(new_data[:, 0].max())):
        per_frame_data1 = new_data[new_data[:, 0] == (i + 1), :]
        if per_frame_data1.shape[0] == 0:
            continue
        task_list1.append(per_frame_data1)

    pool = Pool(processes=cpu_count())
    if int(filename[4]) == 1:
        new_per_frame_data_adding_pf_id_list = pool.map(new_per_frame_data_adding_pf_id, task_list1)
    if int(filename[4]) == 2:
        new_per_frame_data_adding_pf_id_list = pool.map(new_per_frame_data_adding_pf_id, task_list1)
    if int(filename[4]) == 3:
        new_per_frame_data_adding_pf_id_list = pool.map(new_per_frame_data_adding_pf_id, task_list1)
    if int(filename[4]) == 5:
        new_per_frame_data_adding_pf_id_list = pool.map(new_per_frame_data_adding_pf_id, task_list1)
    if int(filename[4]) == 6:
        new_per_frame_data_adding_pf_id_list = pool.map(new_per_frame_data_adding_pf_id, task_list1)
    pool.close()
    pool.join()

    new_data= np.empty((0, 18))
    for new_per_frame_data_adding_pf_id in new_per_frame_data_adding_pf_id_list:
        new_data = np.row_stack((new_data, new_per_frame_data_adding_pf_id))
    time2 = time.time()
    print(str(time2 - time1) + 's')
    print(new_data.shape)
    print('--')

#------------------------------------#
    time3 = time.time()
    task_list2=list()
    for i in range(int(new_data[:, 0].max())):
        per_frame_data2 = new_data[new_data[:, 0] == (i + 1), :]
        if per_frame_data2.shape[0] == 0:
            continue
        task_list2.append(per_frame_data2)

    pool = Pool(processes=(cpu_count()))
    if int(filename[4]) == 1:
        new_per_frame_data_adding_headway_list = pool.map(new_per_frame_data_adding_headway, task_list2)
    if int(filename[4]) == 2:
        new_per_frame_data_adding_headway_list = pool.map(new_per_frame_data_adding_headway, task_list2)
    if int(filename[4]) == 3:
        new_per_frame_data_adding_headway_list = pool.map(new_per_frame_data_adding_headway, task_list2)
    if int(filename[4]) == 5:
        new_per_frame_data_adding_headway_list = pool.map(new_per_frame_data_adding_headway, task_list2)
    if int(filename[4]) == 6:
        new_per_frame_data_adding_headway_list = pool.map(new_per_frame_data_adding_headway, task_list2)
    pool.close()
    pool.join()

    new_data= np.empty((0, 21))
    for new_per_frame_data_adding_headway in new_per_frame_data_adding_headway_list:
        # print(new_per_frame_data_adding_headway.shape)
        new_data = np.row_stack((new_data, new_per_frame_data_adding_headway))
    time4 = time.time()
    print(str(time4 - time3) + 's')
    print(new_data.shape)
    print('---')

    #  新格式：frames, ids, centers_x, centers_y, widths, heights,0-5,
    #        denoised_x, denoised_y, 6-7,, class,8, lane, 9
    #        calculated_vx,calculated_vy,calculated_v, calculated_ax,calculated_ay,calculated_a, 10-12,13-15
    #        preceding_id,following_id,DHW,THW,TTC, 16,17,18-20 共21个数据
    save_path='./02va_headway_extraction_txts/' + filename[:7] + '_va_headway_extraction.txt'
    np.savetxt(save_path, new_data,
               fmt=['%0.0f', '%0.0f', '%0.4f', '%0.4f', '%0.4f', '%0.4f',
                    '%0.4f', '%0.4f', '%0.0f', '%0.0f',
                    '%0.4f', '%0.4f', '%0.4f', '%0.4f', '%0.4f', '%0.4f',
                    '%0.0f', '%0.0f', '%0.4f', '%0.4f', '%0.4f'],delimiter=',')
