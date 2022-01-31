import cv2
import sys
from Display import Display


W = 2160 // 2
H = 3840 // 4

class FeatueExtractor:
    GX = 9
    GY = 8
    def __init__(self):
        self.orb = cv2.ORB_create(100)
        pass

    def extract(self, img):
        kp_all = []
        sy = img.shape[0] // self.GY
        sx = img.shape[1] // self.GX

        for ry in range(0, img.shape[0], sy):
            for rx in range(0, img.shape[1], sx):
                _img = img[ry:ry+sy, rx:rx+sx, :]
                kp = self.orb.detect(_img, None)
                for p in kp:
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    kp_all.append(p)
                    self._drawKpImg(p, img)

    def _drawKpImg(self, kp, img):
        u, v = map(lambda x: int(round(x)), kp.pt)
        cv2.circle(img, (u, v), color=(0, 255, 0), radius=3)
        return


disp = Display(w=W, h=H)
fe = FeatueExtractor()

def process_image(img):
    img = cv2.resize(img, (W, H))
    fe.extract(img)

    # draw only keypoints location,not size and orientation
    # img = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
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