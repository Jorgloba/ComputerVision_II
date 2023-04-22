#!/usr/bin/env python
import sys
import cv2 
import numpy as np
import matplotlib.pyplot as plt

class HistogramHSV:
    def __init__(self, img=None):
        self.img = img

        # originally 8
        self.h_bins = 30
        self.s_bins = 32
        self.hist_size = [self.h_bins, self.s_bins]
        self.h_ranges = [0, 180]
        self.s_ranges = [0, 255]
        self.ranges = self.h_ranges + self.s_ranges
        self.scale=10

    
    def hs_hist1D(self):
        # Create an empty 1D histogram with 180 bins
        hist = np.zeros((180, 1), dtype=np.float32)
        # Convert the image to HSV color space
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        # Get the dimensions of the image
        w, h = self.img.shape[0], self.img.shape[1]
        selection = (0, 0, w, h)
        sel = hue[selection[1]:selection[1] + selection[3], selection[0]:selection[0] + selection[2]]
        # Calculate the histogram
        cv2.calcHist([sel], [0], None, [180], [0, 180], hist, accumulate=0)
        # Normalize the histogram using cv2.normalize()
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        return hist

    def hs_hist2D(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h_plane = hsv[:, :, 0]
        s_plane = hsv[:, :, 1]
        planes = [h_plane, s_plane]

        # Create an empty 2D histogram with the specified number of bins
        hist = np.zeros((self.h_bins, self.s_bins), dtype=np.float32)

        # Calculate the 2D histogram
        #cv2.calcHist(planes, [0, 1], None, hist, self.hist_size, self.ranges, accumulate=0)
        hist = cv2.calcHist(planes, [0, 1], None, self.hist_size, self.ranges, accumulate=0)
        # Normalize the histogram
        max_value = np.max(hist)
        hist /= max_value

        return hist


def plot_hist(hist):
    # Prepare the x-axis (bin edges)
    bins = np.arange(180)
    # Plot the histogram
    plt.bar(bins, hist[:, 0], width=1.0)
    plt.xlabel('Hue')
    plt.ylabel('Frequency')
    plt.title('Hue Histogram')
    plt.show()


def plot_2D_histogram(hist):
    plt.figure()
    plt.imshow(hist, cmap='jet', aspect='auto', origin='lower', interpolation='nearest')
    plt.xlabel('Hue')
    plt.ylabel('Saturation')
    plt.colorbar(label='Frequency')
    plt.title('2D Histogram')
    plt.show()


def hist_comparison(hist1, hist2):
    #print("CORRL=", cv2.compareHist(hist1,hist2, cv2.HISTCMP_CORREL))
    #print("CHISQ=", cv2.compareHist(hist1,hist2, cv2.HISTCMP_CHISQR))
    #print("INTER=", cv2.compareHist(hist1,hist2, cv2.HISTCMP_INTERSECT))
    print("BHATT=", cv2.compareHist(hist1,hist2, cv2.HISTCMP_BHATTACHARYYA))



if __name__ == "__main__":

    src=[]

    src1="image1.jpg"
    src2="image2.jpg"
    src.append(cv2.imread(sys.argv[1]))
    src.append(cv2.imread(sys.argv[2]))

    hsv=[]

    hsv.append(np.zeros(src[0].shape, dtype=np.uint8))
    hsv.append(np.zeros(src[1].shape, dtype=np.uint8)) 


    cv2.namedWindow("im0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("im1", cv2.WINDOW_NORMAL)
    cv2.imshow("im0",src[0])
    cv2.imshow("im1",src[1])
    cv2.waitKey()

    
    # Convert the source images to HSV color space
    cv2.cvtColor(src[0], cv2.COLOR_BGR2HSV, dst=hsv[0])
    cv2.cvtColor(src[1], cv2.COLOR_BGR2HSV, dst=hsv[1])
    cv2.namedWindow("hsv0", cv2.WINDOW_NORMAL)
    cv2.namedWindow("hsv1", cv2.WINDOW_NORMAL)
    cv2.imshow("hsv0",hsv[0])
    cv2.imshow("hsv1",hsv[1])
    cv2.waitKey()
    
    
    # Test 1D and 2D histogram comparisons:
    if (sys.argv[3]=="1D"):
        print("1D analysis")
        hbar=[]
        for i in range(len(src)):
            hbar.append(HistogramHSV( src[i] ))

        hist_comparison(hbar[0].hs_hist1D(), hbar[1].hs_hist1D() )
        for i in range(len(src)):
            hist = hbar[i].hs_hist1D()
            plot_hist(hist)
        


    else:
        print("2D analysis")
        hbar=[]
        for i in range(len(src)):
            hbar.append(HistogramHSV( src[i] ))
            print(hbar[i].hs_hist2D(),  hbar[i].hs_hist2D().shape)

        hist_comparison(hbar[0].hs_hist2D(), hbar[1].hs_hist2D() )
    
        for i in range(len(src)):
            hist=hbar[i].hs_hist2D()
            plot_2D_histogram(hist)
        


