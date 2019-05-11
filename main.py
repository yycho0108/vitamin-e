import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
import numba

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

def local_maxima(img, wsize=11, no_flat=True):
    # curvature
    #img = np.square(img).sum(axis=-1)
    #img = np.linalg.norm(img, axis=-1)
    
    ker = cv2.getStructuringElement(
            cv2.MORPH_RECT, (wsize,wsize) )
    imx = cv2.dilate(img, ker)

    msk = (img >= imx)

    if no_flat:
        e_img = cv2.erode(img, ker)
        flat_msk = (img > e_img)
        msk = np.logical_and(msk, flat_msk)

    idx = np.stack(np.nonzero(msk), axis=-1)

    return msk[...,None].astype(np.float32), idx
    print 'numex', np.sum(msk)
    #return res[..., None]

    #res = np.zeros_like(im)
    #np.copyto(res, im, where=(img >= imx))
    #return res
    #return img * (img >= imx).astype(np.uint8)

class Matcher(object):
    def __init__(self):
        #self.ex_ = cv2.DescriptorExtractor.create("BRIEF")
        self.ex_ = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        self.bf_ = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    @staticmethod
    def to_kpt(idx):
        return [cv2.KeyPoint(e[1],e[0],1) for e in idx]

    def match(self, fr0, fr1, scale=1/2.):
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
        
        dbg = cv2.drawMatches(
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
            pt0 = pt0 / scale
            pt1 = pt1 / scale

        return (pt0, pt1, matches), dbg

def cost_fn():
    pass

def get_dominant_motion(mdata):
    pt0, pt1, m01 = mdata
    i0, i1 = np.stack([(m.queryIdx, m.trainIdx) for m in m01], axis=1)
    #least_squares(cost_fn,
    #cv2.estimateRigidTransform(pt0[i0], pt1[i1], True)
    M, _ = cv2.estimateAffine2D(pt0[i0], pt1[i1])
    A, b = M[:, :2], M[:, 2]
    return A, b

def p_fn(x, sigma=0.1):
    # really rho, but using p anyway
    xsq = np.square(x)
    ssq = np.square(sigma)
    return xsq / (xsq + ssq)

def w_fn(x, sigma=0.1):
    return 1.0 - p_fn(x, sigma=sigma)

#@numba.jit#(nopython=True, parallel=True)
def hill_climb(kappa, pt1, pt1_, F, lmd=0.1):
    # pt1 = (i, j) current position pair
    h, w = kappa.shape[:2]
    
    msk = []
    F1  = []
    pt1_out = []

    for i_p, p in enumerate(pt1):
        i,j = p
        best = F[i_p]
        best_d = None
        keep = False

        for di in [-1,0,1]:
            for dj in [-1,0,1]:
                if di==0 and dj==0:
                    continue
                if i+di < 0 or i+di >= h:
                    continue
                if j+dj < 0 or j+dj >= w:
                    continue
                d_pt = np.linalg.norm(np.float32([i+di, j+dj]) - pt1_[i_p])
                f1 = kappa[i+di, j+dj] + lmd * w_fn(d_pt)
                if f1 > best:
                    best = f1
                    best_d = (di,dj)
                    keep = True

        msk.append(keep)
        if keep:
            di, dj = best_d
            pt1_out.append( (i+di, j+dj) )
            F1.append(best)

    return msk, pt1_out, F1

def hill_climb_vec(kappa, pt1, pt1_, F, lmd=0.1):
    h, w = kappa.shape[:2]

    msk = []
    F1  = []
    pt1_out = []

    best = F
    msk  = np.zeros(len(pt1))

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
        msk, pt1_d, F = hill_climb_vec(kappa, pt1[idx], pt1_[idx], F, lmd=lmd)
        if np.sum(msk) <= 0:
            break
        idx = idx[msk]
        pt1[idx] = pt1_d

    return pt1, good

def main():
    db = []
    matcher = Matcher()

    cam = cv2.VideoCapture(0)
    img = None
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.namedWindow('dbg', cv2.WINDOW_NORMAL)

    trk = None
    while True:
        ret, img = cam.read(img)
        if not ret:
            break
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        kappa = curvature(lab * (100.0/255.0/255.0, 1.0/255.0,1.0/255.0))
        #kappa = normalize(np.abs(kappa), axis=(0,1))
        kappa = np.linalg.norm(kappa, axis=-1)
        #cv2.imshow('kappa', normalize(kappa))
        maxima, idx = local_maxima(normalize(kappa))
        #cv2.imshow('maxima', kappa)
        db.append( (img, idx) )

        if len(db) >= 2:
            mdata, dbg = matcher.match(db[-2], db[-1])
            T_A, T_b = get_dominant_motion(mdata)
            if trk is None:
                trk = db[-2][1]
            trk, good = vitatrack(trk, kappa, T_A, T_b)
            for p in trk:
                cv2.circle(img, (p[1], p[0]), 2, (255,0,0))

            cv2.imshow('dbg', dbg)

        #plt.hist(kappa.ravel(), bins='auto')
        #plt.pause(0.001)

        #print viz.min(axis=(0,1)), viz.max(axis=(0,1))
        #print kappa.min(axis=(0,1)),  kappa.max(axis=(0,1))
        #plt.hist(kappa.ravel())
        #plt.pause(0.001)

        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        luv[..., 2] = 128
        img2 = cv2.cvtColor(luv, cv2.COLOR_LUV2BGR)
        viz = np.clip(img/255.+maxima, 0.0, 1.0)
        #viz = cv2.addWeighted(img, 0.2, kappa, 255./0.8, 0.0,
        #        dtype=cv2.CV_8U)
        #viz = np.concatenate( (img, img2), axis=1)
        #viz = kappa - kappa.min(axis=(0,1),keepdims=True)
        cv2.imshow('img', viz)
        k = cv2.waitKey(1)
        if k in [27, ord('q')]:
            break

if __name__ == '__main__':
    main()
