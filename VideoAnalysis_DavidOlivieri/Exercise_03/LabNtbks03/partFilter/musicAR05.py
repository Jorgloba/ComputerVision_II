#!/usr/bin/env python

import sys
import cv2
import particleFilter03 as pf
import particleModel03 as pm
import histHSV03 as hist
import numpy as np
from operator import itemgetter, attrgetter
import time
#import fluidsynth

img = None
tmp = None
drawing_box = False
num_objects=0
show_regions=False
MAX_OBJECTS=2
region=[]

hand_ref=None

class CvRect:
    def __init__(self,x=None,y=None,width=None,height=None):
        self.x=x
        self.y=y
        self.width=width
        self.height=height
        self.rect=[self.x,self.y,self.width,self.height]


class CvPoint:
    def __init__(self,x=None,y=None,width=None,height=None):
        self.x=x
        self.y=y
        self.point=[x,y]



def draw_box( img, box):
    cv.Rectangle(img, (cv.Round(box.x),cv.Round(box.y)), 
                 (cv.Round(box.x+box.width),cv.Round(box.y+box.height)), 
                 cv.Scalar(0xff,0x00,0x00))
    cv.ShowImage('Camera', img)


def place_box( img, box):
    cv.Rectangle(img, (cv.Round(box.x),cv.Round(box.y)), 
                 (cv.Round(box.x+box.width),cv.Round(box.y+box.height)), 
                 cv.Scalar(0xff,0x00,0x00))
    cv.ShowImage('Camera', img)




def on_mouse( event, x, y, flags, param ):
    global drawing_box
    global num_objects

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
      cv.ShowImage('Camera', img)

      region.append(CvRect())
      region[len(region)-1].x = box.x
      region[len(region)-1].y = box.y
      region[len(region)-1].width  = box.width
      region[len(region)-1].height = box.height


def getROI( img, rg ):
    if ( rg.x+rg.width > img.width or rg.x < 0):
        rg.x=rg.y=0
    if (rg.y+rg.height > img.height or rg.y < 0):
        rg.x=rg.y=0
    
    roi= cv.GetSubRect(img, (rg.x,rg.y,rg.width,rg.height))
    return roi


def connect_nodes( img, r1, r2 ): 
    """
    pt1 = CvPoint(cv.Round(r1.x + r1.width/2), cv.Round( r1.y + r1.height/2))
    pt2 = CvPoint(cv.Round(r2.x + r2.width/2), cv.Round( r2.y + r2.height/2))
    cv.Line(img, pt1, pt2, cv.CV_RGB(0,255,0), 2, 8)
    """
    cv.Line(img, (cv.Round(r1.x + r1.width/2), cv.Round( r1.y + r1.height/2)), 
            (cv.Round(r2.x + r2.width/2), cv.Round( r2.y + r2.height/2)), 
            cv.CV_RGB(0,255,0), 2, 8)
    



def gen_music():
    pass



##  can use a motion detection code (such as camshift) 
##  or optical flow that will key on the position and 
##  put a rectangle around the at the position of maximum 
##  flow...
def self_start():
    pass



def draw_drums(img):
    pt1 =  ( 260, 360 )
    sz =  ( 60, 10 )
    cv.Ellipse(img, pt1, sz, 0,0,360,cv.CV_RGB(0,255,0),2,8)
    x1=cv.Round(pt1[0] - sz[0])
    x2=cv.Round(pt1[0] + sz[0])
    cv.Line(img, (x1,pt1[1]), (x1,pt1[1]+30), cv.CV_RGB(0,255,0), 2, 8)
    cv.Line(img, (x2,pt1[1]), (x2,pt1[1]+30), cv.CV_RGB(0,255,0), 2, 8)
    cv.Ellipse(img, (pt1[0],pt1[1]+30), sz, 0,0,180,cv.CV_RGB(0,255,0),2,8)
    
    
    
    pt2 =  ( 420, 360 )
    sz =  ( 60, 10 )
    cv.Ellipse(img, pt2, sz, 0,0,360,cv.CV_RGB(0,255,0),2,8)
    x1=cv.Round(pt2[0] - sz[0])
    x2=cv.Round(pt2[0] + sz[0])
    cv.Line(img, (x1,pt2[1]), (x1,pt2[1]+30), cv.CV_RGB(0,255,0), 2, 8)
    cv.Line(img, (x2,pt2[1]), (x2,pt2[1]+30), cv.CV_RGB(0,255,0), 2, 8)
    cv.Ellipse(img, (pt2[0],pt2[1]+30), sz, 0,0,180,cv.CV_RGB(0,255,0),2,8)

    return img




def draw_guitar(img):
    # draw the neck
    nb_polylines = 1
    polylines_size = 4
    pt = [0,] * nb_polylines
    for a in range(nb_polylines):
        pt [a] = [0,] * polylines_size
    pt[0][0] = (360,300)
    pt[0][1] = (370,320)
    pt[0][2] = (590,210)
    pt[0][3] = (580,190)

    cv.PolyLine(img, pt, 1, cv.CV_RGB(0,255,0), 2, 8)
    
    return img


def get_handref():
    pass



## ---------------MAIN ----------------------------------

if __name__ == '__main__':
    box = CvRect()
    box.x= box.y = -1
    firstframe=True

    selecthand=False
    time.sleep(2)


    print("pyOpenCv Particle Filter")
    motion = 0
    capture = 0
    if len(sys.argv)==1:
        capture = cv.CreateCameraCapture(0)
    elif len(sys.argv)==2 and sys.argv[1].isdigit():
        capture = cv.CreateCameraCapture(0)
        selecthand=True
        #capture = cv.CreateCameraCapture(int(sys.argv[1]))
    elif len(sys.argv)==2:
        capture = cv.CreateFileCapture(sys.argv[1]) 
        
    if not capture:
        print "Could not initialize capturing..."
        sys.exit(-1)

    """
    frame = cv.QueryFrame(capture)
    frame_size = frame.size()
    fps = capture.get(cv.CV_CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    """
    fps=300
    print fps, 1/fps

    cv.NamedWindow("Camera", 1)
    #cv.NamedWindow("tst1", 1)
    cv.SetMouseCallback( "Camera", on_mouse )

    frames=[]
    ix=0
    Q = pm.ParticleModel()
    roiImg=[]
    new_particles = pf.Particle()
    hbar = None

    genSound=True
    if (genSound):
        fs = fluidsynth.Synth()
        fs.start()

        # cello.
        #sfid = fs.sfload("BHCello.sf2")
        #fs.program_select(0, sfid, 0, 42)

        # drums
        #sfid = fs.sfload("RealAcousticDrums_1.SF2")
        #fs.program_select(0, sfid, 128, 0)

        # guitar
        #sfid = fs.sfload("yamaha.sf2")
        #fs.program_select(0, sfid, 0, 4)

        strum=False
        sfid = fs.sfload("GuitarraAcustica.SF2")
        fs.program_select(0, sfid, 0, 15)

    ncount=0
    sumD=0
    distprev=0
    note=-1
    noteprev=0




    """
    fps= cv.GetCaptureProperty(capture, cv.CV_CAP_PROP_FPS)
    frame = cv.QueryFrame( capture )
    frame_size= cv.GetSize(frame)
    swriter = cv.CreateVideoWriter ("sparticle.avi", 
                                    cv.CV_FOURCC('X','v','i','D'), 
                                    fps, frame_size, True)
    """


    # just wait to establish camera...
    for i in range(100):
        frame = cv.QueryFrame(capture)

    ### ~~~~~~~~ main Loop -~~~~~~~~~~~~~
    while True:
        frame = cv.QueryFrame(capture)
        if(not frame):
            break
        img = cv.CloneImage(frame)
        frames.append(img)

        if (ix == 0):
            w = frames[len(frames)-1].width
            h = frames[len(frames)-1].height

            cv.ShowImage("Camera", frame)

            selectbox=1
            placebox=0
            if (selectbox):
                line_type = cv.CV_AA
                font = cv.InitFont(cv.CV_FONT_HERSHEY_PLAIN, 1, 1, 0.0, 1, line_type)
                text_size, ymin = cv.GetTextSize("OpenCV forever!", font)
                for i in range(0, 601, 100):
                    cv.Line(img, (1,i), (9,i), cv.CV_RGB(0,255,0), 2, 8)
                    pt1 = (10,i)
                    cv.PutText(img, str(i), pt1, font, cv.RGB(0,255,0))

                for i in range(0, 601, 100):
                    cv.Line(img, (i,551), (i,559), cv.CV_RGB(255,0,0), 2, 8)
                    pt1 = (i,560)
                    cv.PutText(img, str(i), pt1, font, cv.RGB(255,0,0))

                #img=draw_drums(img)
                #img=draw_guitar(img)
                #cv.ShowImage("Camera",img)
                #cv.WaitKey()
                #break

                while (num_objects < MAX_OBJECTS):
                    show_regions=True
                    tmp = cv.CloneImage(img)
                    print num_objects
                    if (drawing_box):
                        draw_box(tmp,box)
                    cv.ShowImage("Camera", tmp)
                    cv.WaitKey()

            if (selecthand):
                while (num_objects < MAX_OBJECTS):
                    show_regions=True
                    tmp = cv.CloneImage(img)
                    print num_objects
                    if (drawing_box):
                        draw_box(tmp,box)
                    cv.ShowImage("Camera", tmp)
                    cv.WaitKey()
                placebox=1


            if (placebox):
                box1 = CvRect()
                box2 = CvRect()                
                box1.x=130 
                box1.y=200
                box1.width=box1.height=40

                box2.x=500
                box2.y=250
                box2.width=box2.height=40

                region.append(CvRect())
                region[len(region)-1].x = box1.x
                region[len(region)-1].y = box1.y
                region[len(region)-1].width  = box1.width
                region[len(region)-1].height = box1.height

                region.append(CvRect())
                region[len(region)-1].x = box2.x
                region[len(region)-1].y = box2.y
                region[len(region)-1].width  = box2.width
                region[len(region)-1].height = box2.height
                tmp = cv.CloneImage(img)
                draw_box(tmp,box1)
                draw_box(tmp,box2)
                cv.ShowImage("Camera", tmp)
                cv.WaitKey()


            pnode = []
            for k in range(len(region)):
                print "region k=", region[k].x, region[k].y, region[k].width, region[k].height
                roiImg.append( getROI(img, region[k]) )
                pnode.append ( pf.ParticleNode( k, region[k], roiImg[k]) )
                pnode[k].initialize(50)
                pnode[k].display_particle( img, pnode[k].particles[1] )
                hbar = pnode[k].particles[0].hist
                print "Frame=", ix
                print "---- Region:",pnode[k].particles[0].region

            """
            if ( len(region) > 1 ):
                connect_nodes(img, region[0], region[1] ) 
            cv.ShowImage("Camera", img)            
            """

            img=draw_drums(img)
            cv.ShowImage("Camera", img)
            ts=time.time()
        elif (ix % 1 == 0):
            #cv.WaitKey()
            ## here set up the timer:
            t0 = time.time()
            #print "Track:", ix
            for k in range (len(pnode)):
                #print "----k=", k
                pnode[k].set_pRegions()
                for j in range(pnode[k].np):
                    #print "----j=", j
                    t1 = time.time()
                    pimg = getROI(img, pnode[k].particles[j].region )
                    #print "t1=", time.time() - t1

                    t2 = time.time()
                    pnode[k].particles[j] = Q.transition(pnode[k].particles[j],w,h)

                    #print "t2=", time.time() - t2
                    t3 = time.time()
                    s = pnode[k].particles[j].s
                    #print "t3=", time.time() - t3
                    t4 = time.time()


                    pnode[k].particles[j].w = Q.likelihood( img, 
                                                            cv.Round(pnode[k].particles[j].x), 
                                                            cv.Round(pnode[k].particles[j].y),
                                                            cv.Round(pnode[k].particles[j].width * s),
                                                            cv.Round(pnode[k].particles[j].height * s),  
                                                            pnode[k].particles[j].hist )
                    #print "t4=", time.time() - t4
                    t4 = time.time()

                t5 = time.time()
                pnode[k].normalize_weights()
                new_particles = pnode[k].resample()
                pnode[k].particles = new_particles
                for j in range(pnode[k].np):
                    pnode[k].display_particle( img, pnode[k].particles[j] )



                hbar = pnode[k].particles[0].hist
                #print "t5=", time.time() - t5

            connectline=0
            if ( len(pnode) < MAX_OBJECTS +1 and connectline and genSound):
                ncount= ncount + 1
                connect_nodes(img, pnode[0].particles[0].region, pnode[1].particles[0].region ) 
                dX = cv.Round(pnode[0].particles[0].x) - cv.Round(pnode[1].particles[0].x)
                dY = cv.Round(pnode[0].particles[0].y) - cv.Round(pnode[1].particles[0].y)
                dist = np.sqrt( dX*dX + dY*dY )

                """
                sumD = sumD + dist 
                dist = sumD/ncount
                dist = (dist + distprev)/2
                """

                note = cv.Round( (0.133* dist + 20)   )
                #print "Length=", dist, note 
                if noteprev != note:
                    fs.noteon(0,note, 100)
                    fs.noteoff(0,noteprev)
                noteprev= note
                distprev= dist


            guitarline=1
            if ( guitarline and genSound):
                r1 = pnode[0].particles[0].region
                r2 = pnode[1].particles[0].region
                r3 = CvPoint( 250, 300)
                cv.Line(img, (cv.Round(r1.x + r1.width/2), cv.Round( r1.y + r1.height/2)), 
                        (cv.Round(r3.x), cv.Round( r3.y)), cv.CV_RGB(0,255,0), 2, 8)
                if (r3.y>r2.y):
                    if not strum:
                        fs.noteon(0,56, 100);fs.noteon(0,60,100);fs.noteon(0,63,100)
                        strum=True
                if (r3.y<r2.y):
                    strum=False

            drummodel=0
            if ( drummodel and genSound):
                r1 = pnode[0].particles[0].region
                r2 = pnode[1].particles[0].region

                #img=draw_drums(img)


                print r1.y, r2.y
                if (r1.y > 320): 
                    print "r1.y; note 39"
                    #cv.WaitKey()
                    note=39
                    if (note!=noteprev):
                        fs.noteoff(0,40)
                        fs.noteon(0,39, 100)
                        noteprev= note
                elif (r2.y > 320):
                    print "r2.y; note 40"
                    #cv.WaitKey()
                    note=40
                    if (note!=noteprev):
                        fs.noteoff(0,39)
                        fs.noteon(0,40, 100)
                        noteprev=note

                    

            cv.ShowImage("Camera", img)            
            #cv.WriteFrame(swriter, img)            
            dt =  time.time() - t0        ## difference in wall time.
            epsilon=dt
            #time.sleep(0.25)
            #print "Track=",ix, "time=",time.time()-ts,"dt=", dt, "Length=", dist, "x=",pnode[1].particles[0].x, "y=",pnode[1].particles[0].y, "w=",region[1].width, "h=",region[1].height
        ix=ix+1
        if(cv.WaitKey(10) != -1):
            break
    cv.DestroyWindow("Camera")
