import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from PIL import Image


def draw_points(data,col1,col2,img_path=None):
    if img_path is not None:
        img = Image.open(img_path)
        plt.imshow(img)
    plt.figure(figsize=(9, 6))
    # plt.axis([0, 3840, 0, 2160])
    plt.scatter(data[:,col1],data[:,col2],c='black',s=1,linewidths=1,label='trajectory point')
    plt.title("Trajectory point")
    plt.ylabel("Y(m))")
    plt.xlabel("X(m))")
    plt.legend()
    plt.show()

def draw_xy(data,col1,col2):
    colours = np.random.rand(32, 3)
    plt.figure(figsize=(12, 6))
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        x,y=id_data[:,col1],id_data[:,col2]
        plt.plot(x, y, color=colours[j % 32, :])
        # if i<=3:
        #     plt.plot(x,y, color=colours[j % 32, :],label='trajectory'+' '+str(j))
        # elif i==4:
        #     plt.plot(x,y, color=colours[j % 32, :],label='. . .')
        # else:
        #     plt.plot(x, y, color=colours[j % 32, :])
    plt.title("Trajectory")
    plt.ylabel("Y(pixel)")
    plt.xlabel("X(pixel)")
    # plt.legend(loc="upper left")
    plt.show()

# 绘制3d图
def draw_xy_3d(raw_txt_data):
    data=raw_txt_data
    colours = np.random.rand(32, 3)
    plt.figure(figsize=(10, 7))
    ax = plt.axes(projection="3d")
    for i in range(int(data[:, 1].max())):
        j = i + 1
        id_data = data[data[:, 1] == j, :]
        id_x = id_data[:, 2]
        id_y = id_data[:, 3]
        id_z = id_data[:, 1]

        ax.scatter3D(id_x, id_y, id_z, color=colours[j % 32, :], s=1)
    # ax.set_xticks([])
    ax.set_ylim3d(800, 1600)
    plt.title("simple 3D scatter plot")
    plt.show()

if __name__=='__main__':
    # txt_path ='./outputs/exp2.txt'
    txt_path ='./03dataset_txts/exp0607_dataset.txt'
    data = np.loadtxt(txt_path, delimiter=',',dtype=bytes).astype(str)
    data = data.astype(np.float64)
    # data =data[data[:,24]!=7,:]
    data = data[data[:, 0] <= 9999, :]
    # data = data[data[:, 3] >= 730, :]
    # data = data[data[:, 2] >=1400, :]
    # data=np.row_stack((data33,data53))

    # draw_points(data,2,3)
    # draw_xy(data,6,7)  #6,7
    draw_xy(data,0,14)  #6,7

