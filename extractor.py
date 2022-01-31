import cv2
import numpy as np


class FeatueExtractor:
    GX = 9 #num of rectangles divided widt
    GY = 8 #num of rectangles divided height
    def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.brute_force = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.last = None

    def extract(self, img):
        kps = []
        # detect
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        for feat in feats:
            kps.append(cv2.KeyPoint(x=feat[0][0], y=feat[0][1], _size=20))
            self._drawKpImg(feat, img, color=(0, 0, 255))
        # extract
        kps, des = self.orb.compute(img, kps)
        matches = None
        # matching
        if self.last is not None:
            matches = self.BF_FeatureMatcher(self.last['des'], des)

        self.last = {'kps': kps, 'des': des}
        return feats, des, matches

    def extractORB(self, img):
        kpa = []

        sy = img.shape[0] // self.GY
        sx = img.shape[1] // self.GX
        #
        for ry in range(0, img.shape[0], sy):
            for rx in range(0, img.shape[1], sx):
                _img = img[ry:ry+sy, rx:rx+sx, :]
                kp = self.orb.detect(_img, None)
                for p in kp:
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    kpa.append(p)
                    self._drawKpImg(p, img)
        return kpa

    def BF_FeatureMatcher(self, des1, des2):
        no_of_matches = self.brute_force.match(des1, des2)

        # finding the humming distance of the matches and sorting them
        no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
        return no_of_matches

    def _drawKpImg(self, kp, img, color=(0, 255, 0)):
        u, v = map(lambda x: int(round(x)), kp[0])
        cv2.circle(img, (u, v), color=color, radius=3)
        return
