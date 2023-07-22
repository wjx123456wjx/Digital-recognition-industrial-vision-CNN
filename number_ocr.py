import torch

import LeNET_training as PRE
from LeNET_training import LeNet
from image_preprocess import image_preprocessing


def NumberOCR(image_dir):
    # 设置device gpu
    device = torch.device("cuda:0")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 导入模型
    path = './number_ocr1.pth'  # 需要改为自己的路径
    model = torch.load(path)
    model.to(device)#把模型在gpu上跑
    result = []
    imgs = image_preprocessing(image_dir)#预处理图片
    for i,img in enumerate(imgs):
        inputs = img.reshape(-1, 1, 28, 28)
        # 将数组array转换为张量Tensor
        inputs = torch.from_numpy(inputs)
        inputs = inputs.float()
        inputs = inputs.to(device)

        # 预测
        predict = model(inputs)
        print(predict)
        # 取概率最大的数字的下标:
        print("The number in this picture is {}".format(torch.argmax(predict).detach().cpu().numpy()))
        result.append(torch.argmax(predict).detach().cpu().numpy())
    number = ''
    if len(result)>1:#如果是多为数字
        for i in result:
            number += str(i) #都储存在空字符串中
    else:
        number =  str(result[0])
    print(number)
    return number


if __name__ == '__main__':
    res = NumberOCR('image_archive/3.bmp')