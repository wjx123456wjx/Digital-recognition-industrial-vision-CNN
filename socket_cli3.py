# -*- coding:utf-8 -*-

import socket
import sys

CLI_HOST = 'localhost'
CLI_PORT = 8604
CLI_ADDR = (CLI_HOST, CLI_PORT)
BUFSIZE = 1024

while True:
    sock = socket.socket() # 创建Socket对像
    try:
        a = sock.connect(CLI_ADDR) # 向服务器发出连接请求
    except Exception as e:
        print('error', e)
        sock.close()
        sys.exit()
    else:
        print('have connected with server')
        while True:
            # sendall向服务器发送请求，自动判断每次发送的内容量，删除已发送的部分，剩下的继续传给send()进行发送
            send_data = 'ASKNUM'+'\r'
            sock.sendall(send_data.encode())  # 不要用send()
            recv_data = sock.recv(BUFSIZE)  # 接收请求
            recv_data = recv_data.decode()
            print('receive:',recv_data)  # 解码
            if len(recv_data) > 0:
                print('send:', recv_data)
                # sock.close()
                continue
            else:
                # sock.close()
                continue