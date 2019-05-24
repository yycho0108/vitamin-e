import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import numba
import os

from matcher import Matcher

def curvature(img):
    dfun = cv2.Sobel
    farg = dict(
            ddepth=cv2.CV_64F,
            ksize=5
            )
    #img = img / 255.0 # normalize to 0-1

    fx = dfun(img, dx=1, dy=0, **farg)
    fy = dfun(img, dx=0, dy=1, **farg)
    fxx = dfun(img, dx=2, dy=0, **farg)
    fyy = dfun(img, dx=0, dy=2, **farg)
    fxy = dfun(img, dx=1, dy=1, **farg)

    #fx, fy = dx(img), dy(img)
    #fxx, fyy = dx(img, 2), dy(img, 2)
    #fxy = dy(fx)

    #fx = dfun(img, cv2.CV_64F, 1, 0)
    #fy = dfun(img, cv2.CV_64F, 0, 1)
    #fxx = dfun(img, cv2.CV_64F, 2, 0)
    #fyy = dfun(img, cv2.CV_64F, 0, 2)
    #fxy = dfun(img, cv2.CV_64F, 1, 1)

    k = (fy*fy*fxx - 2*fx*fy*fxy + fx*fx*fyy)
    return k

def normalize(x, vmin=0, vmax=1, axis=-1):
    xmin = np.min(x, axis=axis, keepdims=True)
    xmax = np.max(x, axis=axis, keepdims=True)
    return vmin + (x-xmin) * ((vmax-vmin) / (xmax-xmin))

def local_maxima(img, wsize=9, no_flat=True, thresh=True):
    # local maxima
    ker = cv2.getStructuringElement(
            cv2.MORPH_RECT, (wsize,wsize) )
    imx = cv2.dilate(img, ker)

    msk = (img >= imx)

    if no_flat:
        e_img = cv2.erode(img, ker)
        flat_msk = (img > e_img)
        msk = np.logical_and(msk, flat_msk)

    if thresh:
        val_msk = (img > np.percentile(img, 95.0))
        msk = np.logical_and(msk, val_msk)

    idx = np.stack(np.nonzero(msk), axis=-1)

    return msk[...,None].astype(np.float32), idx

def get_dominant_motion(mdata):
    pt0, pt1, m01 = mdata
    i0, i1 = np.stack([(m.queryIdx, m.trainIdx) for m in m01], axis=1)
    #least_squares(cost_fn,
    #cv2.estimateRigidTransform(pt0[i0], pt1[i1], True)
    M, _ = cv2.estimateAffine2D(pt0[i0], pt1[i1])
    A, b = M[:, :2], M[:, 2]
    return A, b

def p_fn(x, sigma=0.1):
    # really should be `rho`, but using p anyway
    # Geman-McClure Kernel
    xsq = np.square(x)
    ssq = np.square(sigma)
    return xsq / (xsq + ssq)

def w_fn(x, sigma=0.1):
    return 1.0 - p_fn(x, sigma=sigma)

def hill_climb(kappa, pt1, pt1_, F, lmd):
    kappa_pad = np.pad(kappa, ((1,1),(1,1)),
            mode='constant', constant_values=-np.inf)

    Fs = []
    ds = []
    for di in [-1,0,1]:
        for dj in [-1,0,1]:
            ds.append( (di,dj) )
            if di==0 and dj==0:
                Fs.append(F)
                continue
            d_pt = np.linalg.norm(pt1 + [di,dj] - pt1_, axis=-1)
            f = kappa_pad[pt1[:,0]+(1+di), pt1[:,1]+(1+dj)] + lmd * w_fn(d_pt)
            Fs.append(f)

    Fs=np.float32(Fs)
    ds=np.int32(ds)

    sel = np.argmax(Fs, axis=0)
    msk = (sel != 4)
    pt1_out = pt1[msk] + ds[sel[msk]]
    F1  = np.max(Fs,axis=0)[msk]

    return msk, pt1_out, F1

#def lktrack(img0, img1, kpt0):
#    lk_params = dict( winSize  = (15, 15),
#            maxLevel = 2,
#            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
#    pt0 = kpt0[:, ::-1].astype(np.float32) # (i,j) -> (x,y)
#    pt1, _, _ = cv2.calcOpticalFlowPyrLK(img0, img1, pt0, None, **lk_params)
#    pt0r, _, _ = cv2.calcOpticalFlowPyrLK(img1, img0, pt1, None, **lk_params)
#    d = abs(pt0-pt0r).reshape(-1, 2).max(-1)
#    good = d < 4.0
#    pt1 = pt1[:, ::-1].astype(np.int32)
#    return pt1[good], good

def vitatrack(kpt0, kappa, T_A, T_b, lmd=0.001):
    pt0_ = kpt0[:, ::-1].astype(np.float32) # (i,j) -> (x,y)
    pt1_ = pt0_.dot(T_A.T) + T_b # prediction from dominant motion

    pt1  = np.round(pt1_[:,::-1]).astype(np.int32) # back to (i,j) order
    good = np.logical_and.reduce([
        pt1[:,0] >= 0,
        pt1[:,0] < kappa.shape[0],
        pt1[:,1] >= 0,
        pt1[:,1] < kappa.shape[1],
        ])
    pt1_ = pt1_[good]
    pt1  = pt1[good]
    F    = kappa[pt1[:,0], pt1[:,1]]
    idx  = np.arange(len(pt1))

    while True:
        #break
        msk, pt1_d, F = hill_climb(kappa, pt1[idx], pt1_[idx], F, lmd=lmd)
        if np.sum(msk) <= 0:
            break
        idx = idx[msk]
        pt1[idx] = pt1_d

    return pt1, good

class VitaminE(object):
    def __init__(self, lmd=0.001, dbg=True):
        self.matcher_ = Matcher(dbg=dbg)
        self.lmd_  = lmd
        self.db_   = []

        # tracking data
        self.trk_  = None
        self.path_ = []
        self.cols_ = None

        # debugging flag
        self.dbg_ = dbg

        self.reset()

    # properties / parameter setting
    def set_lmd(self, lmd):
        print('lambda : {}'.format(lmd))
        self.lmd_ = lmd

    def reset(self):
        # reset all **data** properties
        self.db_   = []
        self.trk_  = None
        self.path_ = []
        self.cols_ = None

    def __call__(self, img, data={}):
        """ Run Vitamin-E on RGB image """
        kappa = curvature(img / 255.0)
        knorm = np.linalg.norm(kappa, axis=-1)
        max_msk, idx = local_maxima(knorm)
        self.db_.append( (img, idx) )

        if len(self.db_) <= 1:
            return True

        mdata = self.matcher_.match(self.db_[-2], self.db_[-1], scale=1.0, data=data)
        T_A, T_b = get_dominant_motion(mdata)

        if self.trk_ is None:
            # initialize track with initial extrema points
            self.trk_  = self.db_[-2][1]
            self.path_ = self.trk_[None, :]
            self.cols_ = np.random.uniform(0, 255, (len(self.trk_),3))
        self.trk_, good = vitatrack(self.trk_, knorm, T_A, T_b)
        #trk, good = lktrack(db[-2][0], db[-1][0], trk)

        # append + filter data by currently active points
        self.path_ = self.path_[:, good]
        self.cols_ = self.cols_[good]
        self.path_ = np.append(self.path_, self.trk_[None,:], axis=0)

        if self.dbg_:
            # add visualization
            viz = img.copy()
            for p, c in zip(self.path_.swapaxes(0,1)[...,::-1], self.cols_):
                cv2.polylines(viz,
                        #path.swapaxes(0,1)[...,::-1],
                        p[None,...],
                        False, c
                        )
            viz = cv2.addWeighted(img, 0.75, viz, 0.25, 0.0)
            for p, c in zip(self.trk_,self.cols_):
                cv2.circle(viz, (p[1], p[0]), 2, c)

            #viz = cv2.addWeighted(viz, 1.0, max_msk, 255.0, 0.0)
            viz = np.clip(viz + (max_msk * 255), 0, 255).astype(np.uint8)
            data['track-img'] = viz

def main():
    #src = os.path.expanduser('~/Videos/VID_20190327_194904.mp4')
    src = 1
    trk  = None
    path = []
    iter = 0
    scale = 1.

    vita = VitaminE(dbg=True)
    cam = cv2.VideoCapture(src)

    # init gui
    cv2.namedWindow('win', cv2.WINDOW_NORMAL)
    #lset_fn = lambda x: vita.set_lmd(np.log(x-10.0))
    def lset_fn(x):
        vita.set_lmd(np.exp(x-20.0))
    cv2.createTrackbar('lambda', 'win', 10, 20, lset_fn)
    lset_fn(cv2.getTrackbarPos('lambda', 'win'))

    matcher = Matcher()
    img = None

    while True:
        ret, img = cam.read(img)
        if not ret:
            break
        img = cv2.resize(img, None, fx=scale, fy=scale)
        data = {}
        vita(img, data)
        if ('track-img' in data):
            viz = data['track-img']
            #cv2.imwrite('/tmp/frame{:04d}.png'.format(iter), (viz*255).astype(np.uint8) )
            cv2.imshow('win', viz)
        k = cv2.waitKey(1)
        if k in [27, ord('q')]:
            break
        if k in [ord('r')]:
            vita.reset()
        iter += 1

if __name__ == '__main__':
    main()
