import cv2 as cv
import os
import matplotlib.pyplot as plt

img = r"C:\code\cnnt_model\chest_xray"


def imgResize():

    for f in os.listdir(img):
        read_img = cv.imread(img)

        read_img = cv.imshow("", read_img)

        while True:
            wait_key = cv.waitKey(0)
            if wait_key == 27:
                break

imgResize()
