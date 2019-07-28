import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, glob

mask = 0
Q = []  # Queue of seed points found belonging to region. All new points in region are added to it.
        # Corresponding points in mask image are set to 255 to indicate that these points have already been
        # found and marked as belonging to region.

#-------------------------------------------------------------------------------------

def InReg(img, x, y):
    global mask

    if (x < 0 or y < 0 or x >= img.shape[1] or y >= img.shape[0]): return (False)
    # pixel is outside image boundary

    if (img[y,x] > 0):
        if (mask[y][x] == 0):
            mask[y][x] = 255
            return True

    return (False)

# ---------------------------------------------------------------------------

def GrowPt(img, pt):
    global Q
    x, y = pt[0], pt[1]
    try:
        for y in range(pt[1]-1, pt[1]+2):
            for x in range(pt[0]-1, pt[0]+2):
                if InReg(img, x, y):
                    Q.append((x, y))

    except IndexError: pass
    return

# -------------------------------------------------------------------------
def fillGaps(img):
    global mask

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations = 1)

# -------------------------------------------------------------------------
def addSeedPts(img, loc):
    global Q, mask
    mask = np.zeros(img.shape, np.uint8)
    Q.append(loc)
    mask[loc[1], loc[0]] = 255
    return
# -------------------------------------------------------------
DISPLAY = False
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
def GrowRegion(img):
    global mask

    imsize = img.shape
    while (Q != []):
        npt = Q.pop()
        GrowPt(img, npt)    # Finds all neighbours of np and adds them into the list Q, if they are within range

    img2 = img.copy()
    img2 = 0.5 * img2 + 0.5 * mask
    fillGaps(mask)

    showImage(img2, 'Original image', 131)
    showImage(mask, 'mask image', 132)
    showImage(img, 'result image', 133, show=True)

    return(mask)

#--------------------------------------------------------------------------

