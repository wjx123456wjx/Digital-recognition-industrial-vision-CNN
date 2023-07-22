import cv2
import numpy as np

IMAGE_MAX_WIDTH = 1000
IMAGE_MAX_HEIGHT = 1000
MARGIN_RATIO = 6
MIX_TEST_NUM = 5


def image_preprocessing(image_dir, model=None):
    image = []
    img = cv2.imread(image_dir)
    img1 = cv2.resize(img,(1000,1000))
    # cv2.imshow('img',img1)
    # cv2.waitKey()
    # =====================图像处理======================== #
    # 转换成灰度图像
    gray_img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    #二值化处理
    ret, binary = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY_INV)
    # cv2.imshow('binary',binary)
    # cv2.waitKey()
    # #提取外轮廓  cv2.findContours()函数来查找检测物体的轮廓
    contours, hierarchy = cv2.findContours(binary,  # 二值化处理后的选项
                                           cv2.RETR_EXTERNAL,  # 只检测外轮廓
                                           cv2.CHAIN_APPROX_NONE)  # 储存所有的轮廓点
    #print('contours[0].shape', len(contours))
    #print(contours)
    # 返回轮廓定点及边长
    borders = []
    for contour in contours:
        #返回四个值，分别是x，y，w，h；x，y是矩阵左上点的坐标，w，h是矩阵的宽和高
        x, y, w, h = cv2.boundingRect(contour)  # len(contours)-2
        #print('x', x, 'y', y, 'w', w, 'h', h)
        #如果轮廓太小，则不保存
        if w * h > 5000:
            borde = [(x, y), (x + w, y + h)]
            borders.append(borde)
    #给识别到的数字轮廓排序：先按行排，然后按列排
    for m in range(len(borders)):
        for n in range(m + 1, len(borders)):
            if borders[m][1][1] > borders[n][1][1]:
                temp = borders[n]
                # print(temp)
                borders[n] = borders[m]
                borders[m] = temp
    for m in range(len(borders)):
        for n in range(m + 1, len(borders)):
            if borders[n][1][1] - borders[m][1][1] < 300 and borders[m][0][0] > borders[n][0][0]:
                temp = borders[n]
                # print(temp)
                borders[n] = borders[m]
                borders[m] = temp
    #print(borders)
    #根据轮廓绘制方框
    for i, border in enumerate(borders):
        cv2.rectangle(gray_img,
                      border[0],  # 轮廓列表
                      border[1],  # 绘制全部轮廓
                      (0, 0, 255),  # 轮廓颜色
                      2)  # 轮廓粗细
        #根据方框的位置依次在一开始的灰度图上进行裁剪
        borderImg = gray_img[border[0][1] + 2:border[1][1] - 2, border[0][0] + 2:border[1][0] - 2]
        # cv2.imshow('borderImg{}'.format(i), borderImg)
        # cv2.waitKey()
        # 高斯去噪
        gauss_img = cv2.GaussianBlur(borderImg, (5, 5), 0, 0, cv2.BORDER_DEFAULT)
        # cv2.imshow('gauss_img', gauss_img)
        # cv2.waitKey()
        # 将图像扩展到正方形
        dst_size = (28, 28)
        src_h, src_w = gauss_img.shape[:2]
        #print(src_h, src_w)
        dst_h, dst_w = dst_size
        # 判断应该按哪个边做等比缩放
        h = dst_w * (float(src_h) / src_w)  # 按照ｗ做等比缩放
        w = dst_h * (float(src_w) / src_h)  # 按照h做等比缩放
        h = int(h)
        w = int(w)
        if h <= dst_h:
            image_dst = cv2.resize(gauss_img, (dst_w, int(h)))
        else:
            image_dst = cv2.resize(gauss_img, (int(w), dst_h))
        h_, w_ = image_dst.shape[:2]
        #print(h_, w_)
        top = int((dst_h - h_) / 2);
        down = int((dst_h - h_ + 1) / 2);
        left = int((dst_w - w_) / 2);
        right = int((dst_w - w_ + 1) / 2);

        value = [255, 255, 255]
        borderType = cv2.BORDER_CONSTANT
        #print(top, down, left, right)

        #填充成原比例的正方形
        image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)
        # cv2.imshow('image_dst', image_dst)
        # cv2.waitKey()

        # 获取图像的高和宽
        high = image_dst.shape[0]
        wide = image_dst.shape[1]

        # 将图像每个点的灰度值进行阈值比较
        for h in range(high):
            for w in range(wide):
                # 若灰度值大于100，则判断为背景并赋值0，否则将深灰度值变白处理
                if image_dst[h][w] > 100:
                    image_dst[h][w] = 0
                else:
                    image_dst[h][w] = 255 #- image_dst[h][w]
        #cv2.imshow('result', image_dst)
        #cv2.waitKey()
        #把处理好的图片放到集合中
        image.append(image_dst)
        #print(len(image))
    cv2.waitKey()
    cv2.destroyAllWindows()
    return image

if __name__ == '__main__':

    image_preprocessing('record_image/Tq.bmp')


