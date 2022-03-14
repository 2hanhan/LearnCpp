import sys
import cv2


def Hello(s):
    print("Hello World")
    print(s)


def Add(a, b):
    print('a=', a)
    print('b=', b)
    return a + b


class Test:
    def __init__(self):
        print("Init")

    def SayHello(self, name, old):
        print("Hello,", name, old)
        return name


def show_image(name):
    print(name)
    image = cv2.imread(name)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    return name
