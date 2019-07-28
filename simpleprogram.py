import cv2
import matplotlib.pyplot as plt
import numpy as np
import math
import os
import pydicom
from pydicom.data import get_testdata_files
import FindRegion as FR
import scipy.ndimage.filters as fi
#plt.set_cmap('gray')    # to set default colour for matplotlib to gray

# -------------------------------------------------------------------------
path = 'C:\\Users\\shubh\\Desktop\\Studies\\Image Processing Project\\'
#file = 'HFXMP5JU.bmp'
file = 'Image1.bmp'
#-----------------------------------------------------------------------------

DISPLAY = True
BLOCK = False
import sys      # to check if the program is running in debug mode or not
def showImage(im, title=None, numWin = None, show=False):
    if 'pydevd' in sys.modules and DISPLAY:     # The variable pydevd is true if program is running in debug mode
        if numWin != None: plt.subplot(numWin)
        plt.imshow(im)
        plt.title(title)
        if show:
            plt.show(BLOCK)
            plt.waitforbuttonpress()
            plt.close()
            return

#-------------------------------------------------------------------

def templateMatching(img, tpl):    #It uses numpy array multiplication and summation which is very fast
    ims = img.shape
    ts = tpl.shape
    ts = ts[0]//2, ts[1]//2
    corIm = np.zeros(ims, int)
    for r in range(ts[0], ims[0]-ts[0]):
        for c in range(ts[1], ims[1]-ts[1]):
            subim = img[r-ts[0]:r+ts[0]+1, c-ts[1]:c+ts[1]+1]
            corr = np.sum(subim * tpl)
            corIm[r, c] = corr
    return (corIm)

#----------------------------------------------------------------------------
def plotLineHisto(line, width=None):   #Summaation of pixels in a sub-window of width X width in the image is available in corSum
    plt.plot(line), plt.show(block=False)
    plt.text(50, 50, 'width = {}'.format(width))
    plt.waitforbuttonpress()
    plt.close()

#----------------------------------------------------------------------------

def findMaxLoc(cimg):     #regSum is summaation of pixels in a sub-window of width X width in the image for all pixels in image
    h, w = cimg.shape
    max = int(cimg.max())
    maxIdx = np.argmax(cimg)
    rmax = maxIdx // w
    cmax = maxIdx - rmax * w

    return((cmax, rmax), max)

#----------------------------------------------------------------------------
# it finds sum of all positive correlation values in a sub-window of increasing size till it reaches peak value
def findMaxRegion(cimg):
    h, w = cimg.shape[0:2]
    max = cimg.max()
    maxloc = np.argmax(cimg)
    rmax = maxloc // w
    cmax = maxloc - rmax * w
    return ((cmax, rmax), max)

#----------------------------------------------------------------------------
def showResult(img, tpl, loc):
    global DISPLAY
    tpl -= np.min(tpl)
    tpl = (tpl * 255) // np.max(tpl)
    h, w  = tpl.shape    #size of template after resizing that was found in image
    x, y = loc[0] - w//2, loc[1]-h//2 #location of template
    alpha = 0.75
    resIm = img.copy()
    for r in range(h):
        for c in range(w):
            resIm[y+r, x+c] = alpha * resIm[y+r, x+c] + (1-alpha) * tpl[r,c]
    DISPLAY = True
    showImage(resIm, 'template superimposed', show=True)

#--------------------------------------------------------------------------------

def activeDisc(r1=20):
    r2 = int(math.sqrt(2) * r1 + 2)
    tsize = 2*r2+1
    adisc = np.zeros((tsize, tsize), np.int8)
    for r in range(-tsize//2, (tsize+1)//2):
        for c in range(-tsize // 2, (tsize + 1) // 2):
            rad = math.sqrt(r**2+c**2)
            if rad <= r1:
                adisc[tsize//2+r, tsize//2+c] = -1
            elif rad <= r2:
                adisc[tsize // 2 + r, tsize // 2 + c] = 1
    return (adisc)

#--------------------------------------------------------------------------------
def LoGDisc(radedge, sigma=3, plot=False):
    gsize = sigma * 3
    rad = radedge + 4*sigma
    circularEdge = np.ones((2 * rad + 1, 2 * rad + 1), float)

    gaussian = np.zeros((2 * gsize + 1, 2 * gsize + 1), float)
    gaussian [ gsize, gsize] = 1
    gaussian = fi.gaussian_filter ( gaussian, sigma)
    if plot: showImage(gaussian, 'Gaussian', show=True)
    if plot: plotLineHisto(gaussian[gsize])
    LoGTpl = cv2.Laplacian(gaussian, cv2.CV_64F, ksize=3)
    if plot: showImage(LoGTpl, 'LoG', show=True)
    if plot: plotLineHisto(LoGTpl[gsize])

    for ri in range(-rad, rad+1):
        for ci in range(-rad, rad + 1):
            r2 =  math.sqrt(ri*ri + ci*ci)
            if r2 > radedge:
                circularEdge[ri+rad, ci+rad] = 0

    template = cv2.filter2D(circularEdge, -1, LoGTpl)
    if plot: showImage(template, 'LoG', show=True)
    if plot: plotLineHisto(template[rad])
    return(template)

#--------------------------------------------------------------------
def readImage(file):
    if file[-4:] == '.dcm':
        dimg = pydicom.dcmread(file)   # read the dicom image
        imgf = dimg.pixel_array     # extract the pixel data from the dicom image file
        img = np.uint8(imgf * (255 / np.max(imgf)))  # to create ndarray of bytes needed for displaying image
    else:
        img = cv2.imread(file, 0)
        img = cv2.resize(img, (256, 256))
    return (img)
#--------------------------------------------------------------------------------
def main():
    global DISPLAY
    rho, sigma = 20, 4
    img = readImage(path+file)
    LoGTpl = LoGDisc(rho, sigma)
    showImage(img, "image", 121)
    showImage(LoGTpl, "template", 122, True)
    #DISPLAY = False

    curmax, maxloc = 0, (0,0)
    for s in range (-2, 3):
        LoGTpl = LoGDisc(rho+s, sigma)
        CorImg = templateMatching(img, LoGTpl)
        loc, max = findMaxRegion(CorImg)
        output = "CorImg with s = {} %d at loc {}: and max is {}".format(s, loc, max)
        #print(output)
        showImage(CorImg, output, show=True)
        if max > curmax:
            curmax = max
            maxloc = loc
            maxs = s
            print('current max and location is for s ', max, loc, maxs)
    #----------------------

    print (maxloc)
    LoGTpl = LoGDisc(rho+maxs, sigma)
    showResult(img, LoGTpl, loc)

#----------------------------------------------------------------------------
main()