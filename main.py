import cv2
import sys
from Display import Display


W = 2160 // 2
H = 3840 // 2

disp = Display(w=W, h=H)

def process_image(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (W, H))
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