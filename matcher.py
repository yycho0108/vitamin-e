import cv2
import numpy as np

class Matcher(object):
    def __init__(self, dbg=True):
        # params
        self.dbg_ = dbg

        # handles
        #self.ex_ = cv2.DescriptorExtractor.create("BRIEF")
        self.ex_ = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.bf_ = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @staticmethod
    def to_kpt(idx):
        return [cv2.KeyPoint(e[1],e[0],1) for e in idx]

    def match(self, fr0, fr1, scale=1/2., data={}):
        # WARNING: kpt0 is index, NOT cv2.KeyPoint()
        img0, kpt0 = fr0
        img1, kpt1 = fr1

        if scale is None: scale=1.0

        if (scale != 1.0):
            img0 = cv2.resize(img0, dsize=None, fx=scale, fy=scale)
            kpt0 = (kpt0 * scale)#.astype(np.int32)
            img1 = cv2.resize(img1, dsize=None, fx=scale, fy=scale)
            kpt1 = (kpt1 * scale)#.astype(np.int32)

        kpt0 = Matcher.to_kpt(kpt0)
        kpt1 = Matcher.to_kpt(kpt1)
        kpt0, des0 = self.ex_.compute(img0, kpt0)
        kpt1, des1 = self.ex_.compute(img1, kpt1)
        matches = self.bf_.match(des0, des1)
        matches = sorted(matches, key=lambda x:x.distance)

        if self.dbg_:
            data['match-img'] = cv2.drawMatches(
                    img0,kpt0,
                    img1,kpt1,
                    matches[:],
                    None,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )

        pt0 = cv2.KeyPoint.convert(kpt0)
        pt1 = cv2.KeyPoint.convert(kpt1)

        if (scale != 1.0):
            # undo scale transform
            pt0 = np.divide(pt0, scale)
            pt1 = np.divide(pt1, scale)

        return (pt0, pt1, matches)
