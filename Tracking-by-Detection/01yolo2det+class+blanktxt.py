import shutil

from PIL import Image
import os
import glob
import numpy as np


# 读取图片，修改图片，另存图片
def convertjpg(jpgfile, outdir, img_sum=0):
    img = Image.open(jpgfile)  # 提取目录下所有图片
    try:
        img_sum = str('%08d' % img_sum)  # 图片保存成00000001格式
        img.save(os.path.join(outdir) + img_sum + '.jpg')  # 保存到另一目录
    except Exception as e:
        print(e)

# 读取文件名
def file_name(file_dir):
    L = []  # 保存文件名
    img_num = 0  # 计算图片总数
    for root, dirs, files in os.walk(file_dir):
        img_num = img_num + len(files)
        one = os.path.basename(str(root))  # 获取路径最后的/或者\后的文件名
        L.append(one)
    num = len(L) - 1  # 获取路径下文件个数
    print('%s路径下有%d个文件' % (L[0], num))
    return L, num, img_num

def txt2det(txt_dir, out_dir, txt_num):
    temp=np.array([[txt_num,-1,0,0,60,30,0.999,-1,-1,-1,0]])
    if not os.path.exists(txt_dir):
        np.savetxt(out_dir, temp, fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d", delimiter="\n")
    else:
        data = np.loadtxt(txt_dir, dtype='float')
        # print(data.shape)
        # for k in data[:, 0]:
            # print(k)
        data = data.reshape(-1, 6)
        cls=np.zeros(len(data[:,0]))
        cls[:]=data[:,0]
        data[:, 0] = txt_num
        data[:, 1] = (data[:, 1] - 0.5 * data[:, 3]) * 3840  # 左上角坐标x
        data[:, 2] = (data[:, 2] - 0.5 * data[:, 4]) * 2160  # 左上角坐标y
        data[:, 3] = (data[:, 1] + data[:, 3] * 3840)  # 右下角坐标x
        data[:, 4] = (data[:, 2] + data[:, 4] * 2160)  # 右下角坐标y
        a = np.linspace(-1, -1, len(data))
        data = np.insert(data, 1, a, axis=1)  # 行矩阵插入
        a = a.reshape((len(a), 1))
        data = np.concatenate((data, a, a, a, a), axis=1)  # 补充-1
        data[:,-1]=cls

        data = np.row_stack((data, temp))####手动添加的边框
        np.savetxt(out_dir, data, fmt="%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%d,%d,%d", delimiter="\n")

def main():
    # for txtfile in sorted(glob.glob('D:/07Experiment/05MOT-by-yolov5/01yolov5-02-txt/exp3/labels/' + '*.txt'),key=os.path.getsize):
    #     print(txtfile)
    #     i = i + 1
    #     print(i)
    #     txt2det(txtfile, 'D:/07Experiment/05MOT-by-yolov5/01yolov5-02-txt/exp3/dets/' + str(i) + '.txt', i)

    ## 修改 ##
    # root_path='E:/07Experiment-data/VideoSets/02AerialVideos-04TextFiles/01expressway/exp0107/'
    # root_path='E:/07Experiment-data/VideoSets/03AerialVideos-VehicleVerificationTexts/exp0607/exp2/'
    root_path='E:/07Experiment-data/VideoSets/02AerialVideos-05LongTextFiles/exp0503/exp5/'
    common_name_in_labels='exp0503-right_masked_'

    txt_dir=root_path + '/labels/'
    output_dir=root_path + '/dets/'
    output_final_det=root_path + '/det.txt'

    num=len(glob.glob(txt_dir + '*.txt'))
    # print(glob.glob(txt_dir + '*.txt'))
    for n in glob.glob(txt_dir + '*.txt'):
        n_ = int(n.split('_')[-1].split('.')[0])
        if num < n_:
            num = n_
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    for i in range(num):
        i=i+1
        print(i)

        txt2det(txt_dir + common_name_in_labels +str(i)+'.txt',
                output_dir + str(i) + '.txt', i)

    det = open(output_final_det, 'w')
    for txts in range(num):
        print(txts+1)
        one_det = open(output_dir + str(txts + 1) + '.txt').read()
        det.write(one_det)
    det.close()
    # shutil.copy(output_final_det,'./data/train/ADL-Rundle-6/det/det.txt')
    print('det.txt生成完成')

if __name__ == '__main__':
    main()


