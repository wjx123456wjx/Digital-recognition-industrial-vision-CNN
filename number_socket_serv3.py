# -*- coding:utf-8 -*-

import socket
from number_ocr import NumberOCR
from LeNET_training import LeNet
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from number_ocrui4 import Ui_number_ocr
import sys
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtWidgets import QMainWindow, QTextEdit, QAction, QApplication
import os,sys
from PyQt5.QtCore import Qt
import torch
import numpy as np
from number_ocr import NumberOCR
from LeNET_training import LeNet
import cv2 as cv
import GxAcquireSoftTrigger as GxCam
BUFSIZE = 1024

sub_threads = []


class window(QtWidgets.QMainWindow,Ui_number_ocr):
    def __init__(self):
        super(window, self).__init__()
        self.HOST = ''
        self.PORT = 8604
        self.ADDR = (self.HOST, self.PORT)
        self.BUFSIZE = 1024
        self.listen_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        # listen_sock.settimeout(5.0)  # 设定超时时间后，socket其实内部变成了非阻塞，但有一个超时时间
        self.listen_sock.bind(self.ADDR)  # 将socket对象server绑定到ADDR对应的地址上。ADDR为双元素的元组tuple，其中两个元素依次为ip与端口号。该操作为服务端确定了一个地址。
        self.listen_sock.listen(2)  # 最大连接数量
        self.connect_sock =None
        self.client_addr = None
        self.data = None
        self.cwd = os.getcwd()
        self.setupUi(self)
        self.labels = self.imageBrowser
        self.img=None
        self.result = None
    #开启进程
    def start_thread(self):
        print('build connect when new TCP comes')
        self.connect_sock, self.client_addr = self.listen_sock.accept()  # socket对象server开始监听 #接受客户端连接
        self.data = self.connect_sock.recv(BUFSIZE)  # 接收数据
        if self.connect_sock:
            self.textBrowser.append('%s from fanuc robot 连接成功' % self.client_addr[0])
        else:
            self.textBrowser.append('%s from fanuc robot 连接失败' % self.client_addr[0])
    #打开文件夹选择图片
    def slot_open_image(self):
        file, filetype = QFileDialog.getOpenFileName(self, '打开多个图片', self.cwd, "*.jpg, *.png, *.JPG, *.JPEG, All Files(*)")
        jpg = QtGui.QPixmap(file).scaled(self.labels.width(), self.labels.height())
        self.labels.setPixmap(jpg)
        self.img=file
        self.textBrowser.append('选择图片成功')
    #拍照
    def photograph(self):
        if self.data.decode() == 'ASKNUM'+'\r':
            self.textBrowser.append('正在拍照中。。。。')
            num = 'test'
            image_name = GxCam.cam_trigger(str(num))
            file = 'D:/PycharmProjects/numberorc/record_image/' + image_name + '.bmp'
            print('图片路径：',file)
            jpg = QtGui.QPixmap(file).scaled(self.labels.width(), self.labels.height())
            self.labels.setPixmap(jpg)
            self.img = file
            # order = str(1) + '\r'
            # print('receive order:', order)
            # self.connect_sock.sendall(order.encode())
            # print('send feedback:', order)
        else:
            self.textBrowser.append('机器人命令无法识别，请退出系统重启！')

    #识别图片
    def slot_output_digital(self):
        self.textBrowser.append('正在识别中。。。。')
        path =self.img
        path = '//'.join(path.split('/'))
        result1 = NumberOCR(path)
        if len(result1) > 0:
            self.textBrowser.append('识别成功！')
        else:
            self.textBrowser.append('识别失败！请检查图片是否正确')
        self.numberBrowser.setText(result1)
        self.result = result1
    #发送数据
    def send_data(self):

        if self.data.decode() == 'ASKNUM'+'\r':
            self.textBrowser.append('正在发送中。。。。')
            while len(self.result) < 4:
                self.result = '0' + self.result
            if len(self.result) > 4:
                print("数字过多，取前4位")
                print(self.result)
                self.result = self.result[:4]
            # print('receive order:', data)
            for i,number in enumerate(self.result):
                order = str(number)+ '\r'
                print('order:', order)
                self.connect_sock.sendall(order.encode())
                print('send 第{}个数字'.format(i+1), number)
                # self.textBrowser.append('send 第{}个数字'.format(i + 1), number)
                self.textBrowser.append('发送第{}个数字成功！'.format(i+1))
                self.data = self.connect_sock.recv(BUFSIZE)  # 接收数据
                print(self.data.decode())
                # while not self.data.decode() == 'NEXNUM' +'\r':
                #
                #     continue
        else:
            self.textBrowser.append('机器人命令无法识别，请关闭线程重启！')
    #关闭线程
    def close_thread(self):
        if self.connect_sock:
            self.connect_sock.close()
            self.textBrowser.append('关闭线程成功!')
        else:
            self.textBrowser.append('没有找到线程')


if __name__ == "__main__":
  app = QtWidgets.QApplication(sys.argv)
  my = window()
  my.show()
  sys.exit(app.exec_())