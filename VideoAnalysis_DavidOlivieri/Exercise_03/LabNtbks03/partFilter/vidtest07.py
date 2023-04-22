#!/usr/bin/env python

##  D.Olivieri: (updated 2-may-2010)
##    *python based particle filter tracking code.
##    * module namespace specific.
##    * incorporates the latest code for histograms,     
##      and other improvements. 

import sys
import pyopencv as cv
import particleFilter as pf
import particleModel as pm
import histHSV as hist
import numpy as np
from operator import itemgetter, attrgetter

img = None
tmp = None
drawing_box = False
num_objects=0
show_regions=False
MAX_OBJECTS=2
region=[]


def draw_box( img, box):
    cv.rectangle(img, cv.Point(box.x,box.y), cv.Point(box.x+box.width,box.y+box.height), cv.Scalar(0xff,0x00,0x00))
    cv.imshow('Camera', img)
 
def on_mouse( event, x, y, flags, param ):
    global drawing_box
    global num_objects

    if img.empty():
        return;


    if( event == cv.CV_EVENT_LBUTTONDOWN ):
        drawing_box=True
        box.x=x
        box.y=y
        box.width=0
        box.height=0
    elif( event == cv.CV_EVENT_MOUSEMOVE and (flags & cv.CV_EVENT_FLAG_LBUTTON) ):
        if (drawing_box):
            box.width  = x - box.x
            box.height = y - box.y
            #print box
            #imshow('Camera', img)

    elif( event== cv.CV_EVENT_LBUTTONUP): 
      drawing_box = False;
      if (num_objects >= MAX_OBJECTS):
          return
      if( box.width<0  ):
          box.x+=box.width  
          box.width *=-1
      if( box.height<0 ):
          box.y+=box.height 
          box.height*=-1 
      draw_box( img, box );
      num_objects = num_objects+1
      print num_objects
      cv.imshow('Camera', img)

      region.append(cv.CvRect())
      region[len(region)-1].x = box.x
      region[len(region)-1].y = box.y
      region[len(region)-1].width  = box.width
      region[len(region)-1].height = box.height
      

def getROI( img, rg ):
    #print "getROI: ", rg.x, rg.y, rg.width, rg.height
    xrange=cv.Range(rg.x, rg.x+rg.width)
    yrange=cv.Range(rg.y, rg.y+rg.height)
    roi= cv.Mat(img, yrange, xrange)
    return roi



def connect_nodes( img, r1, r2 ): 
    pt1 = cv.Point(cv.round(r1.x + r1.width/2), cv.round( r1.y + r1.height/2))
    pt2 = cv.Point(cv.round(r2.x + r2.width/2), cv.round( r2.y + r2.height/2))
    cv.line(img, pt1, pt2, cv.CV_RGB(0,255,0), 2, 8)

## ---------------MAIN ----------------------------------

if __name__ == '__main__':
    box = cv.CvRect()
    box.x= box.y = -1
    firstframe=True


    print("pyOpenCv Particle Filter")
    cv.namedWindow('Camera', cv.CV_WINDOW_AUTOSIZE)
    cv.moveWindow('Camera', 10, 10)
    try:
        device = int(sys.argv [1])
        del sys.argv [1]
    except (IndexError, ValueError):
        device = 0
    if len (sys.argv) == 1:
        capture = cv.VideoCapture(device)
    else:
        capture = cv.VideoCapture(sys.argv [1])            

    if not capture.isOpened():
        print("Error opening capture device")
        sys.exit (1)

    frame = cv.Mat()
    capture >> frame
    frame_size = frame.size()
    fps = capture.get(cv.CV_CAP_PROP_FPS)

    if fps == 0:
        fps = 30
    writer = cv.VideoWriter ("captured.avi",cv.CV_FOURCC('X','v','i','D'), fps, frame_size, True)

    if not writer.isOpened():
        print("Error opening writer")
        sys.exit (1)


    print("Before the loop")
    cv.setMouseCallback( "Camera", on_mouse )




    frames=[]
    ix=0
    Q = pm.ParticleModel()
    roiImg=[]
    new_particles = pf.Particle()

    hbar =hist.HistogramHSV()

    ### ~~~~~~~~ main Loop -~~~~~~~~~~~~~
    while True:
        capture >> frame
        if frame.empty():
            break
        img = frame.clone()
        frames.append(frame.clone())


        if (ix == 0):
            print "Frame Dim:",frames[len(frames)-1].cols, frames[len(frames)-1].rows
            w = frames[len(frames)-1].cols
            h = frames[len(frames)-1].rows

            while (num_objects < MAX_OBJECTS):
                show_regions=True
                tmp = img.clone()
                print num_objects
                if (drawing_box):
                    draw_box(tmp,box)
                cv.imshow("Camera", tmp)
                cv.waitKey()

            if show_regions:
                show_regions=False
                for i in region:
                    print i

            # create the nodes at selected regions.
            pnode = []
            for k in range(len(region)):
                print region[k]
                roiImg.append( getROI(img, region[k]) )
                pnode.append ( pf.ParticleNode( k, region[k], roiImg[k]) )
                pnode[k].initialize(15)
                pnode[k].display_particle( img, pnode[k].particles[1] )

                hbar = pnode[k].particles[0].hist
                #hbar.printHist(ix)
                print "Frame=", ix, hbar.size
                print "---- Region:",pnode[k].particles[0].region

                for jx in range(hbar.size):
                    if hbar.histo[jx] != 0:
                        print jx, hbar.histo[jx]

            if ( len(region) > 1 ):
                connect_nodes(img, region[0], region[1] ) 
            cv.imshow("Camera", img)            
            

        else:
            ## this continues to draw the rectangles...
            print "Track:", ix
            for k in range (len(pnode)):
                ##print "index:", k, "length:", pnode[k].particles[0].hist.size, \
                ##    "x=",pnode[k].particles[0].x, "width=",pnode[k].particles[0].width
                pnode[k].set_pRegions()
                print "region, particles[0]=", pnode[k].particles[0].region
                ##print "region, particles[1]=", pnode[k].particles[1].region


                for j in range(pnode[k].np):
                    pimg = getROI(img, pnode[k].particles[j].region )
                    pnode[k].particles[j] = Q.transition(pnode[k].particles[j],w,h)
                    s = pnode[k].particles[j].s
                    #print "...particles:", cv.round(pnode[k].particles[j].x), cv.round(pnode[k].particles[j].y)
                    #pnode[k].particles[j].w = Q.likelihood( pimg, pnode[k].particles[j].hist )
                    pnode[k].particles[j].w = Q.likelihood( img, 
                                                            cv.round(pnode[k].particles[j].x), 
                                                            cv.round(pnode[k].particles[j].y),
                                                            cv.round(pnode[k].particles[j].width * s),
                                                            cv.round(pnode[k].particles[j].height * s),  
                                                            pnode[k].particles[j].hist )


                    #print "Weights=", k,j, pnode[k].particles[j].w
                pnode[k].normalize_weights()
                new_particles = pnode[k].resample()
                pnode[k].particles = new_particles
                

                for j in range(pnode[k].np):
                    pnode[k].display_particle( img, pnode[k].particles[j] )

                hbar = pnode[k].particles[0].hist
                #hbar.printHist(ix)
                """
                print "Frame=", ix, hbar.size
                for jx in range(hbar.size):
                    if hbar.histo[jx] != 0:
                        print jx, hbar.histo[jx]
                """
               

            if ( len(pnode) < MAX_OBJECTS +1 ):
                connect_nodes(img, pnode[0].particles[0].region, pnode[1].particles[0].region ) 
                ##connect_nodes(img, pnode[1].particles[0].region, pnode[2].particles[0].region ) 
                """
                connect_nodes(img, pnode[2].particles[0].region, pnode[3].particles[0].region ) 
                connect_nodes(img, pnode[3].particles[0].region, pnode[4].particles[0].region ) 
                connect_nodes(img, pnode[4].particles[0].region, pnode[5].particles[0].region ) 
                connect_nodes(img, pnode[5].particles[0].region, pnode[6].particles[0].region ) 
                connect_nodes(img, pnode[3].particles[0].region, pnode[7].particles[0].region ) 
                """

            cv.imshow("Camera", img)            
            #cv.waitKey()

        ix=ix+1
        if (cv.waitKey(15) == 27): 
            break
        
        #~~~~~~~~~  End main loop


