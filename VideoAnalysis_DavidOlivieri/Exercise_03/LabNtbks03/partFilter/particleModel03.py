#!/usr/bin/env python
import sys
import cv2 as cv
import particleFilter03 as pf
import histHSV03 as hist
import numpy as np

## ----------------GLOBALS ----------------
TRANS_X_STD=1.0   ## original 1.0
TRANS_Y_STD=1.0   ## original 0.5
TRANS_S_STD=0.001
A1 = 2.0    #original 2.0
A2 = -1.5   #original -1.0
B0 = 4.000  # original 1.0  (better with 2.0)
LAMBDA = 500  # original 20
#--------------------------------------


def getROI(img, rg):
    if rg.x + rg.width > img.shape[1] or rg.x < 0:
        rg.x = rg.y = 0
    if rg.y + rg.height > img.shape[0] or rg.y < 0:
        rg.x = rg.y = 0

class CvRect:
    def __init__(self,x=None,y=None,width=None,height=None):
        self.x=x
        self.y=y
        self.width=width
        self.height=height
        self.rect=[self.x,self.y,self.width,self.height]

class ParticleModel:
    def __init__(self, dynamics=None):    
        self.dynamics = dynamics
        print("TRANS_X_STD=", TRANS_X_STD)
        print("TRANS_Y_STD=", TRANS_Y_STD) 
        print("TRANS_S_STD=", TRANS_S_STD)
        print("A1=", A1) 
        print("A2=", A2)   
        print("B0=", B0)  
        print("LAMBDA=", LAMBDA)  

    def likelihood(self, img, px, py, w, h,  hst):
        rg = CvRect()
        rg.x = cv.round(px - w/2)
        rg.y = cv.round(py - h/2) 
        rg.width = w
        rg.height = h
        roi = getROI(img, rg)
        h = hist.HistogramHSV( roi )
        hPrime= h.hs_hist2D()
        cv.imshow("rg",roi)
        dsq = self.dist_metric(hPrime, hst, 4)
        return np.exp( - LAMBDA * dsq )

    def dist_metric(self, h1, h2, method):
        dist=0;
        if (method==1):
            dist=cv.compareHist(h1,h2, cv.HISTCMP_CORREL)
        elif (method==2):
            dist=cv.compareHist(h1,h2, cv.HISTCMP_CHISQR)
        elif (method==3):
            dist= cv.compareHist(h1,h2, cv.HISTCMP_INTERSECT)
        elif (method==4):
            dist=cv.compareHist(h1,h2, cv.HISTCMP_BHATTACHARYYA)
        return dist

    def transition(self, p, w, h):
        pn = pf.Particle()
        x = A1*( p.x - p.x0 ) + A2 * ( p.xp - p.x0 ) + B0 * TRANS_X_STD* np.random.randn() + p.x0
        pn.x = x
        y = A1 * ( p.y - p.y0 ) + A2 * ( p.yp - p.y0 ) + B0 * TRANS_Y_STD * np.random.randn() + p.y0
        pn.y = y
        s = A1 * ( p.s - 1.0 ) + A2 * ( p.sp - 1.0 ) + B0 * TRANS_S_STD * np.random.randn() + 1.0
    
        pn.xp = p.x
        pn.yp = p.y
        pn.sp = p.s
        pn.x0 = (p.xp + p.x)/2.0
        pn.y0 = (p.yp + p.y)/2.0
        pn.width = p.width
        pn.height = p.height
        pn.hist = p.hist
        pn.w = 0
        rg = CvRect()
        rg.x = cv.Round(p.x - p.width/2)
        rg.y = cv.Round(p.y - p.height/2) 
        rg.width = p.width
        rg.height = p.height
        pn.region = rg

        return pn


## ----- MAIN for Testing -----------------

if __name__ == '__main__':
    print("particleModel class")
    region=[]
    pnode=[]
    Q = ParticleModel()
    for k in range(3):
        region.append(CvRect())
        region[k].x = int( 100 * np.random.rand() )
        region[k].y= int(100 * np.random.rand())
        region[k].width=100
        region[k].height=100
        print(region[k])
        pnode.append ( pf.ParticleNode( k, region[k] ) )
        pnode[k].initialize(5)

    for k in range(len(pnode)):
        for j in range(pnode[k].np):
            pnode[k].particles[j] =  Q.transition( pnode[k].particles[j], region[k].width, region[k].height )



