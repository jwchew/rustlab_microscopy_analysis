#/usr/bin/python

"""
Update log:

November 15, 2016
-----------------
Added capability to interface with the cell selection program I wrote that
generates bwImg files for each brightfield image instead of letting the
program do the cell segmentation on its own.  This is accomplished by first
running the cell segmentation program to segment the cells and produce
"segmented.npy" files that are masks demarcating each cell.  The modified
script below loads bwImg from the "segmented.npy" file instead of performing
its own cell segmentation as before.

February 13, 2018
-----------------
Removed the dependence for the sequence of images to be continuous in
numbering, thus allowing for experiments in which images are not recorded in
the dark but where the image numbers are still incremented.  Specifically,
each image that is read in uses the empirically defined scan num list instead
of just using the numbers in the range() function, which assumes continuous
image numbering.  Additionally, the trP output is modified to use the actual
timepoint data with gaps for darkness when images were not acquired.
"""

import cv2
import os
import numpy as np
import celltracker
import sys
cdir = os.getcwd()

# CHANGE SUFFIX BASED ON BRIGHTFIELD FILE NAME
#suffix = "590nm.tif"
suffix = "550nm.tif"

if __name__ == '__main__':
    fPath = sys.argv[1]
    if len(sys.argv) > 2:
        nSteps = sys.argv[2]
    else:
        nSteps = 300
else:
    fPath = (raw_input(("Please enter the folder you want to analyze: ")))
    nSteps = (raw_input(("Please enter the number of"
                         "steps you want to analyze: ")))

def processFiles(input_tuple):
    
    # allows for multiple arguments to be fed into pool.map()
    dirID = input_tuple[0]
    numSteps = input_tuple[1]
    
    AA = []
    LL = []
    masterList = []
    regionP = []
    bwL = []
    bwL0 = []
    k = 0
    fileList = celltracker._sort_nicely(os.listdir(dirID))
    tifList = [f for f in fileList if "segmented-rescaled" in f]
    scan_nums = [f.split("_")[2] for f in tifList]
    """ # Old code that tried to process all images in the folder
    endsWith = suffix
    tifList = []
    for f in fileList:
        if f.endswith(endsWith):
            tifList.append(f)
    """
    posID = tifList[0].split('_')[4]
    if os.path.exists(dirID+'/translationList.txt'):
        tList = np.loadtxt(dirID+'/translationList.txt')
    else:
        tList = celltracker.stabilizeImages(dirID, suffix,
                                            SAVE=False, preProcess=True)
        np.savetxt('translationList.txt', tList)
    tList = tList.astype('int')
    for i in range(np.min((len(tifList)-2, numSteps))):
        fooB = cv2.imread((dirID+'/WT_scan_'+scan_nums[i]+'_num_'+posID +
                           '_Brightfield-' + suffix), -1)
        fooB = np.roll(np.roll(fooB, tList[i][0], axis=0),
                       tList[i][1], axis=1)
        fooC = cv2.imread((dirID+'/WT_scan_'+scan_nums[i]+'_num_'+posID +
                           '_Chlorophyll.tif'), -1)
        fooC = cv2.medianBlur(fooC.copy(), 3)
        fooC = np.roll(np.roll(fooC, tList[i][0], axis=0),
                       tList[i][1], axis=1)

        fooY = cv2.imread((dirID+'/WT_scan_'+scan_nums[i]+'_num_'+posID +
                           '_YFP.tif'), -1)
        fooY = cv2.medianBlur(fooY.copy(), 3)
        fooY = np.roll(np.roll(fooY, tList[i][0], axis=0),
                       tList[i][1], axis=1)

#        preBW = celltracker.preProcessCyano(fooB, fooC, mask=True)
#        bwDist = cv2.distanceTransform(preBW, cv2.cv.CV_DIST_L2, 5)
#        bwImg = celltracker.processImage(bwDist.copy(),
#                                         boxSize=51, solidThres=0)
        bwImg = np.load(dirID+'/WT_scan_'+scan_nums[i]+'_num_'+posID +
                            '_segmented.npy')
        bwImg = np.roll(np.roll(bwImg, tList[i][0], axis=0),
                       tList[i][1], axis=1)

        bwL = celltracker.bwlabel(bwImg.copy())
        regionP = celltracker.regionprops(bwImg.copy(), 1)
        cv2.imwrite(dirID+'/WT_scan_'+scan_nums[i]+'_num_'+posID+'_processed.tif',
                    np.uint8(bwL))
        print dirID, i

        if regionP.size:
            if (k > 0):
                areaList = celltracker.labelOverlap(bwL0, bwL)
                AA.append(areaList)
                linkList = celltracker.matchIndices(areaList, 'Area')
                LL.append(linkList)
            bwL0 = bwL.copy()
            avgCellY = celltracker._avgCellInt(fooY, bwImg.copy())
            if np.isnan(avgCellY).any():
                avgCellY[np.isnan(avgCellY)] = 0
            avgCellC = celltracker._avgCellInt(fooC, bwImg.copy())
            if np.isnan(avgCellC).any():
                avgCellC[np.isnan(avgCellC)] = 0
            regionP = np.hstack((regionP, avgCellY[1:]))
        #   regionP[:, 4]=avgCellC[1:].reshape(-1)
            masterList.append(regionP)
            k = k+1

    celltracker.saveListOfArrays(dirID+'/masterL.npz', masterList)
    celltracker.saveListOfArrays(dirID+'/LL.npz', LL)
#    try:
    tr = celltracker.linkTracks(masterList,  LL)
    trP = celltracker.processTracks(tr)
    # replace time column in trP with actual timepoints instead of sequential numbering
    int_scan_nums = np.array([int(f) for f in scan_nums], dtype=np.float64)
    print trP.shape
    trP[:,2] = int_scan_nums[trP[:,2].astype(int)]
    print trP.shape
    np.save(dirID+'/trP', trP)
#    except:
#        print "Warning: Could not process tracks for", dirID

print fPath
processFiles((fPath, nSteps))
