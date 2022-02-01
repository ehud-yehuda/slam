import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

fEst = []

class FeatueExtractor:
    GX = 9 #num of rectangles divided widt
    GY = 8 #num of rectangles divided height
    def __init__(self, K, show=True):
        self.orb = cv2.ORB_create(100)
        self.brute_force = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None
        self.show = show
        self.K = K
        self.Kinv = np.linalg.inv(self.K)

    def denormlizeCoords(self, data):
        data = np.append(data, np.ones((data.shape[0], 1)), axis=1)
        return self.K.dot(data.transpose()).transpose()[:, :-1]

    def normlizeCoords(self, data):
        # data[:, :, 0] -= img_shape[0] // 2
        # data[:, :, 1] -= img_shape[1] // 2
        data = np.append(data, np.ones((data.shape[0], 1)), axis=1)
        return self.Kinv.dot(data.transpose()).transpose()[:, :-1]

    def extract(self, img):
        kps = []
        # detect
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        for feat in feats:
            kps.append(cv2.KeyPoint(x=feat[0][0], y=feat[0][1], _size=20))
            # self._drawKpImg(feat, img, color=(0, 0, 255))

        # extract
        kps, des = self.orb.compute(img, kps)

        # matching
        matches = None
        RT = None
        if self.last is not None:
            matches = self.BF_knnMatch(kps, des, img)

            #filter
            matches, model = self.filterMatching(matches, img.shape)
            RT = self.estimateRT(model)
            if self.show:
                for match in matches:
                    self._drawMatchesImg(match=match, img=img)

        self.last = {'kps': kps, 'des': des}
        print(RT)
        return matches, RT

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

    def BF_FeatureMatcher(self, des1, des2, img):
        no_of_matches = self.brute_force.match(des1, des2)

        # finding the humming distance of the matches and sorting them
        no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
        matches = zip([kps[m.queryIdx] for m in matches], [kps[m.trainIdx] for m in matches])
        [self._drawMatchesImg(img, m) for m in matches]
        return no_of_matches

    def BF_knnMatch(self, kps, des, img):
        matches = self.brute_force.knnMatch(des, self.last['des'], k=2)
        ret = []
        for m,n in matches:
            if m.distance < 0.75 * n.distance:
                kp1 = kps[m.queryIdx].pt
                kp2 = self.last['kps'][m.trainIdx].pt
                ret.append((kp1, kp2))
        return ret

    def filterMatching(self, data, img_shape):
        if data is not None and len(data) > 0:
            data = np.array(data)
            #normalize coords movin to center
            data[:, 0, :] = self.normlizeCoords(data[:, 0, :])
            data[:, 1, :] = self.normlizeCoords(data[:, 1, :])

            model, inliers = ransac((data[:, 0], data[:, 1]),
                                    # FundamentalMatrixTransform,
                                    EssentialMatrixTransform,
                                    min_samples=8,
                                    # residual_threshold=1,
                                    residual_threshold=0.001,
                                    max_trials=100)
            data = data[inliers]
            #denormalized the coords back
            data[:, 0, :] = self.denormlizeCoords(data[:, 0, :])
            data[:, 1, :] = self.denormlizeCoords(data[:, 1, :])
        return data, model

    def estimateRT(self, model):
        u, sig, vt = np.linalg.svd(model.params)
        # done with fundemental matrix model residual higher
        # f_est = np.sqrt(2) / ((v[0] + v[1])/2)
        # fEst.append(f_est)
        # print(f_est, np.median(fEst))

        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        assert np.linalg.det(u) > 0
        if np.linalg.det(vt) < 0:
            vt *= -1.0

        R = np.dot(np.dot(u, W), vt)
        if np.sum(R.diagonal()) < 0:
            R = np.dot(np.dot(u, W.transpose()), vt)
        T = u[:, 2]
        return np.append(R, T.reshape(3, 1), axis=1)


    def _drawKpImg(self, kp, img, color=(0, 255, 0)):
        # u, v = map(lambda x: int(round(x)), kp)
        u, v = int(round(kp[0])), int(round(kp[1]))
        cv2.circle(img, (u, v), color=color, radius=3)
        return u, v

    def _drawMatchesImg(self, img, match):
        u1, v1 = self._drawKpImg(match[0], img)
        u2, v2 = self._drawKpImg(match[1], img)
        cv2.line(img, (u1, v1), (u2, v2), (255, 0, 0))
        return