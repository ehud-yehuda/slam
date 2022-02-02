

class Frame(object):
    def __init__(self, img, extractor):
        self.extractor = extractor
        self.K = extractor.K
        self.Kinv = extractor.Kinv
        self.kp, self.des, self.RT = self.extractor.extract(img)
        return
