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
path = 'C:\\AKGupta\\DSI\\PYTHON36\\PythonProj\\Myocardium\\Images\\CarotidArtery\\'
#file = 'HFXMP5JU.bmp'
file = 'EMGNV3P3.bmp'
#-----------------------------------------------------------------------------

DISPLAY = True
import sys      # to check if the program is running in debug mode or not
def showImage(im, title=None, numWin = None, show=False):
    if 'pydevd' in sys.modules and DISPLAY:     # The variable pydevd is true if program is running in debug mode
        if numWin != None: plt.subplot(numWin)
        plt.imshow(im)
        plt.title(title)
        if show:
            plt.show(block=False)
            plt.waitforbuttonpress()
            plt.close()
            return

#-------------------------------------------------------------------

def templateMatching2(img, tpl):    #It uses numpy array multiplication and summation which is very fast
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

def findMaxLoc(regSum):     #regSum is summaation of pixels in a sub-window of width X width in the image for all pixels in image
    h, w = regSum.shape
    max = int(regSum.max())
    regSum[regSum < int(max*0.75)] = 0

    maxIdx = np.argmax(regSum)
    rmax = maxIdx // w
    cmax = maxIdx - rmax * w

    FR.addSeedPts(regSum, (cmax, rmax))
    reg = FR.GrowRegion(regSum)
    return((cmax, rmax), reg)

#----------------------------------------------------------------------------
# it finds sum of all positive correlation values in a sub-window of increasing size till it reaches peak value
def findMaxRegion(cimg):
    h, w = cimg.shape[0:2]
    cimg[cimg<500] = 0    # negative or low correlation values are ignored by setting them to zero
    lineSum = np.array(cimg, float) # It will store the sum of all pixels (on a line) in the range (col +/- width)
    regSum = np.array(cimg, float) # this will store the sum of all pixels in a square sub-window of size 2*width+1
    # we look for local max value in this image

    for s in range(1, 20):
        for r in range(h):
            for c in range(s, w-s):
                lineSum[r][c] += cimg[r][c-s] + cimg[r][c+s]
                # It sums all pixels on the line corIm[r] in the range (c +/- s) and stores the result in corIm[r][c]
            #plotLineHisto(lineSum[r], s)
        regSum[:] = 0
        for r in range(s, h-s):  # now sum lines in the range (r-s to r+s)
            for i in range(-s, s+1):
                regSum[r] += lineSum[r+i]
        loc, reg = findMaxLoc(regSum)

        reg[reg>250] = 100  # So that intensity of region goes down and boundary drawn as white line becomes visible
        showImage(reg, 'Region detected', show=True)
        image, conList, hierarchy = cv2.findContours(reg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print('number of contours found is {}'.format(len(conList)))
        for cnt in conList:
            cv2.drawContours(reg, [cnt], 0, (255, 255, 255), 1)
        showImage(reg, 'location is {} '.format(loc), show=True)
        if len(conList) == 1:
            radius = math.sqrt(cv2.contourArea(cnt) / 3.1416)
            arcLen = cv2.arcLength(cnt, True)
            ratio = arcLen / radius
            print (radius, arcLen, ratio)
            if ratio < 8:   # the ideal value is 2 * pi
                return (loc, s)
    return ((-1, -1), -1)   # no location found

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
def readImage(file, DICOM=True):
    if DICOM:
        dimg = pydicom.dcmread(file)   # read the dicom image
        imgf = dimg.pixel_array     # extract the pixel data from the dicom image file
        img = np.uint8(imgf * (255 / np.max(imgf)))  # to create ndarray of bytes needed for displaying image
    else:
        img = cv2.imread(file, 0)
        img = cv2.resize(img, (256, 256))
    return (img)
#--------------------------------------------------------------------------------
def main():
    r1, sigma = 18, 4
    template = activeDisc(20)     #LoGDisc(r1, sigma, plot=False)   # activeDisc()
    #To display the template and its plot, make plot to True
    img = readImage(path+file, DICOM=False)
    CorImg = templateMatching2(img, template)

    showImage(img, "Image", 121)
    showImage(template, "Laplacian of Gaussian", 122, show=True)
    showImage(CorImg, "Correlation image", show=True)

    CorImg[CorImg < 500] = 0
    loc, s = findMaxRegion(CorImg)  # gives location of the template and its scale
    print (loc)
    h,w = template.shape
    template = LoGDisc(r1, sigma)#activeDisc(r1+s-1, sigma)
    showResult(img, template, loc)

#----------------------------------------------------------------------------
main()