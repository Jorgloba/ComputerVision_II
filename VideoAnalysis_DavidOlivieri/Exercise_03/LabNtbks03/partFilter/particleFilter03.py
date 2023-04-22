#!/usr/bin/env python
import sys
import cv2
import numpy as np
import histHSV03 as hist
from operator import attrgetter, itemgetter

class CvRect:
    def __init__(self,x=None,y=None,width=None,height=None):
        self.x=x
        self.y=y
        self.width=width
        self.height=height
        self.rect=[x,y,width,height]

class CvPoint:
    def __init__(self,x=None,y=None,width=None,height=None):
        self.x=x
        self.y=y
        self.point=[x,y]

class Particle:
    def __init__(self):
        self.x=0         ## current x coordinate 
        self.y=0         ## current y coordinate 
        self.s=1.0         ## scale
        self.xp=0        ## previous x coordinate 
        self.yp=0        ## previous y coordinate 
        self.sp=0        ## previous scale 
        self.x0=0        ## original x coordinate 
        self.y0=0        ## original y coordinate 
        self.width=0     ## original width of region described by particle
        self.height=0    ## original height of region described by particle
        self.hist=None     ## histogram...
        self.w=0         ## weight 
        self.region=CvRect()      ## each will carry a region....

    def getROI(self):
        pass

class ParticleNode:
    def __init__(self, index, region=None, roi=None):
        self.index = index
        self.region = region  ## this should be the reference region
        self.particles = []
        self.ref_hist = None
        self.np=10
        self.roi = roi   ## bring the image to the particle node for calcuating
                         ## the histograms
        print("in ParticleNode Constructor ---->", region)

    def initialize(self, np):
        h = hist.HistogramHSV( self.roi )
        self.ref_hist = h.hs_hist2D()

        self.np = np
        for i in range(np):
            self.particles.append( Particle() )

        x = self.region.x + self.region.width // 2
        y = self.region.y + self.region.height // 2
        for k in range(np):
            self.particles[k].x0 = self.particles[k].xp = self.particles[k].x = x;
            self.particles[k].y0 = self.particles[k].yp = self.particles[k].y = y;
            self.particles[k].sp = self.particles[k].s = 1.0;
            self.particles[k].width = self.region.width;
            self.particles[k].height = self.region.height;
            self.particles[k].w = 0;
            self.particles[k].region = self.region
            self.particles[k].hist = self.ref_hist

    def normalize_weights(self):
        sum = 0.0
        for i in range(self.np):
            sum += self.particles[i].w
        inv_sum = 1.0/sum
        for i in range(self.np):
            self.particles[i].w *= inv_sum

 

    def resample(self):
        new_particles = []
        for i in range(self.np):
            new_particles.append( Particle() )

        self.particles = sorted(self.particles, key=attrgetter('w'), reverse=True)
        k=0
        for i in range(self.np):
            np = cv2.round(self.particles[i].w * self.np) 
            for j in range(np):
                new_particles[k] = self.particles[i]
                k+=1
                if (k==self.np): 
                    return new_particles
        while (k < self.np):
            new_particles[k] = self.particles[i]
            k+=1

        return new_particles


    def display_particle(self, img, p):
        x0 = round( p.x - 0.5 * p.s * p.width ) 
        y0 = round( p.y - 0.5 * p.s * p.height )
        x1 = x0 + round( p.s * p.width ) 
        y1 = y0 + round( p.s * p.height )
        cv2.rectangle(img, (x0 ,y0), (x1,y1), (0, 0, 255))

    def get_mostLikely(self):
        self.particles.sort(key=lambda p: p.w, reverse=True)
        return self.particles[0]


        
    def set_pRegions(self):
        for k in range(self.np):
            x = self.particles[k].x
            y = self.particles[k].y
            width  = self.particles[k].width
            height = self.particles[k].height
            self.particles[k].region.x = cv2.Round( x - width/2 )
            self.particles[k].region.y = cv2.Round( y - height/2 )
            self.particles[k].region.width  = width
            self.particles[k].region.height = height



if __name__ == '__main__':
    region=[]
    pnode=[]
    for k in range(3):
        region.append(CvRect())
        region[k].x = int( 100 * np.random.rand() )
        region[k].y= int(100 * np.random.rand())
        region[k].width=100
        region[k].height=100
        print(region[k].x, region[k].y)

        pnode.append ( ParticleNode( k, region[k] ) )
        pnode[k].initialize(5)