import cv2 as cv
from cv2 import aruco
import numpy as np
from math import pi,atan2,asin

def getRotation(R):
    roll = 180*atan2(-R[2][1], R[2][2])/pi
    pitch = 180*asin(R[2][0])/pi
    yaw = 180*atan2(-R[1][0], R[0][0])/pi
    rotation= [roll,pitch,yaw]
    return rotation

calibrationPath = "calibration/Matrix.npz"

calibrationData = np.load(calibrationPath)

cameraMatrix =calibrationData["camMatrix"]
distCoef = calibrationData["distanceCoef"]
size = 6 #cm

dictionary = aruco.Dictionary_get(aruco.DICT_4X4_1000)
markers = aruco.DetectorParameters_create()

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    bwFrame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    markerCorners,markerID,reject = aruco.detectMarkers(
        bwFrame, dictionary, parameters = markers
    )
    if markerCorners:
        rVec,tVec, _ = aruco.estimatePoseSingleMarkers(markerCorners,size,cameraMatrix,distCoef)
        totalMarkers = range(0,markerID.size)
        for ID, corners,i in zip(markerID,markerCorners, totalMarkers):
            cv.polylines(frame,[corners.astype(np.int32)],True,(142,255,255),4,cv.LINE_AA)
            corners = corners.reshape(4,2)
            corners = corners.astype(int)
            topRight = tuple(corners[0])
            bottomLeft = tuple(corners[3])
            bottomRight = tuple(corners[2])
            rotation,_ = cv.Rodrigues(rVec[i])
            rotation = getRotation(rotation)

            point = cv.drawFrameAxes(frame,cameraMatrix,distCoef,rVec[i],tVec[i],5,5)
            cv.putText(
                frame,
                f"ID: {ID[0]} Distance: {round(tVec[i][0][2],2)}",
                topRight,
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (255,255,0),
                2,
                cv.LINE_AA,
                False
            )
            cv.putText(
                frame,
                f"x: {round(tVec[i][0][0],2)} y: {round(tVec[i][0][1],2)}",
                bottomLeft,
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (255,255,0),
                2,
                cv.LINE_AA,
                False
            )
            cv.putText(
                frame,
                f"Roll: {round(rotation[0],2)} Pitch: {round(rotation[1],2)} Yaw: {round(rotation[2],2)}",
                bottomRight,
                cv.FONT_HERSHEY_COMPLEX,
                1,
                (255,255,0),
                2,
                cv.LINE_AA,
                False
            )


    cv.imshow("frame",frame)
    key = cv.waitKey(1)
    if key == ord('e'):
        break
cap.release()
cv.destroyAllWindows()