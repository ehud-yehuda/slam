import cv2
import sys

import numpy as np

#my classes
from Display import Display
from extractor import FeatueExtractor
from Frame import Frame

W = 2160 // 2
H = 3840 // 4

fx, fy = 432.0, 432.0
K = np.array([[fx, 0, W // 2], [0, fy, H // 2], [0, 0, 1]])

disp = Display(w=W, h=H)
fe = FeatueExtractor(K)


frames = []
def process_image(img):
    img = cv2.resize(img, (W, H))
    frames.append(Frame(img, fe))
    disp.draw(img)

def run():
    path = '/home/udi/workspace/slam/data/test_drone.mp4'
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_image(frame)
            # pass
        else:
            break

    return 0

if __name__ == '__main__':
    sys.exit(run())