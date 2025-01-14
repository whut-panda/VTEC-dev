import cv2
import numpy as np
from tqdm import tqdm
import argparse
import os

if __name__ == '__main__':
    # __input_path  = 'E:/07Experiment-data/VideoSets/02AerialVideos-01合并后视频/Stitching202310261200-H220-4K30fps_left-1_199+200+201.mp4'
    # __output_path = 'E:/07Experiment-data/VideoSets/02AerialVideos-01合并后视频/Stitching202310261200-H220-4K30fps_left-1_199+200+201_stb.mp4'
    __input_path  = 'D:/lyz/DJI_20241122192604_0005_V.MP4'
    __output_path = 'D:/lyz/DJI_20241122192604_0005_V_stb.mp4'
    __number = -1

    ########## video_capture
    __capture = {'cap': cv2.VideoCapture(__input_path), 'size': None, 'frame_count': None, 'fps': None, 'video': None}
    __capture['size'] = (int(__capture['cap'].get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(__capture['cap'].get(cv2.CAP_PROP_FRAME_HEIGHT)))
    __capture['fps'] = __capture['cap'].get(cv2.CAP_PROP_FPS)
    __capture['video'] = cv2.VideoWriter(__output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         __capture['fps'], __capture['size'])
    __capture['frame_count'] = int(__capture['cap'].get(cv2.CAP_PROP_FRAME_COUNT))
    if __number == -1:
        __number = __capture['frame_count']
    else:
        __number = min(__number, __capture['frame_count'])

    ########### 初始化surf
    # surf 特征提取
    __surf = {
        'surf': None, # surf算法
        'kp': None, # 提取的特征点
        'des': None, # 描述符
        'template_kp': None # 过滤后的特征模板
    }
    # 配置
    __config = {
        # 要保留的最佳特征的数量
        'key_point_count': 5000,
        # Flann特征匹配
        'index_params': dict(algorithm=0, trees=5),
        'search_params': dict(checks=50),
        'ratio': 0.5,
    }
    __capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
    state, first_frame = __capture['cap'].read()
    __capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, __capture['frame_count'] - 1)
    state, last_frame = __capture['cap'].read()

    __surf['surf'] = cv2.xfeatures2d.SURF_create(__config['key_point_count'])
    __surf['kp'], __surf['des'] = __surf['surf'].detectAndCompute(first_frame, None)
    kp, des = __surf['surf'].detectAndCompute(last_frame, None)

    flann = cv2.FlannBasedMatcher(__config['index_params'], __config['search_params'])
    matches = flann.knnMatch(__surf['des'], des, k=2)
    good_match = []
    for m, n in matches:
        if m.distance < __config['ratio'] * n.distance:
            good_match.append(m)
    __surf['template_kp'] = []
    for f in good_match:
        __surf['template_kp'].append(__surf['kp'][f.queryIdx])



    ########### 处理
    current_frame = 1
    __capture['cap'].set(cv2.CAP_PROP_POS_FRAMES, 0)
    process_bar = tqdm(__number, position=current_frame)
    while current_frame <= __number:
        success, frame = __capture['cap'].read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 计算特征点
        kp, des = __surf['surf'].detectAndCompute(frame_gray, None)
        # 快速临近匹配
        flann = cv2.FlannBasedMatcher(__config['index_params'], __config['search_params'])
        matches = flann.knnMatch(__surf['des'], des, k=2)
        # 计算单应性矩阵
        good_match = []
        for m, n in matches:
            if m.distance < __config['ratio'] * n.distance:
                good_match.append(m)

        # 特征模版过滤
        p1, p2 = [], []
        for f in good_match:
            # print(f.queryIdx)
            # print(f.trainIdx)
            if __surf['kp'][f.queryIdx] in __surf['template_kp']:
                p1.append(__surf['kp'][f.queryIdx].pt)
                p2.append(kp[f.trainIdx].pt)


        # 单应性矩阵
        H, _ = cv2.findHomography(np.float32(p2), np.float32(p1), cv2.RHO)
        # 透视变换
        output_frame = cv2.warpPerspective(frame, H, __capture['size'], borderMode=cv2.BORDER_REPLICATE)

        # 写帧
        __capture['video'].write(output_frame)
        current_frame += 1
        process_bar.update(1)

    # 释放
    __capture['video'].release()
    __capture['cap'].release()
