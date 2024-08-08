import cv2 as cv
import os
import numpy as np

chessBoardDimensions = (9,6)
size = 25 #mm

directory = "calibration"

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)

obj3D = np.zeros((chessBoardDimensions[0]*chessBoardDimensions[1],3),np.float32)

obj3D[:,:2] = np.mgrid[0:chessBoardDimensions[0],0 : chessBoardDimensions[1]].T.reshape(-1,2)

obj3D *=size

objectPoints = []
imagePointts = []

imageDir = "images"

files = os.listdir(imageDir)

for file in files:
    imagePath = os.path.join(imageDir,file)

    image = cv.imread(imagePath)
    bwImage = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    ret,corners = cv.findChessboardCorners(bwImage,chessBoardDimensions)
    if ret == True:
        objectPoints.append(obj3D)
        corners2 = cv.cornerSubPix(bwImage,corners,(3,3),(-1,-1),criteria)
        imagePointts.append(corners2)

cv.destroyAllWindows()
ret, matrix,distance,rotationVectos,TranslationVector = cv.calibrateCamera(
    objectPoints,imagePointts,bwImage.shape[::-1],None,None
)

print("Saving")
np.savez(
    f"{directory}/Matrix",
    camMatrix = matrix,
    distanceCoef=distance,
    rotationVector=rotationVectos,
    translationVector=TranslationVector,
)
