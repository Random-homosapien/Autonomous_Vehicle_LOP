import numpy as np
import cv2 as cv
import glob

#FIND CHESSBOARD AND CREATE POINTS

##CHECK##
chessboard_Size = (10,7)
frameSize = (640,480) #Size of camera Frame

# Termination criteria
criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

#prepare obj points
objp = np.zeros((chessboard_Size[0] * chessboard_Size[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboard_Size[0], 0:chessboard_Size[1]].T.reshape(-1,2)

##CHECK##
objp = objp * 20 # as there is 20mm gap between 2 points in chessboard
print(objp)

# Arrays to store obj and img points
objpoints = []
imgpointsL = []
imgpointsR = []

imagesLeft = glob.glob('stereoLeft/*.png')
imagesRight = glob.glob('stereoRight/*.png')
print("Hello")
for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    print ("Reading img")
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    #Find chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboard_Size, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboard_Size, None) # or cv.findChessboardCornersSB

    # If found, add obj points, img points 
    if retL and retR == True:
        objpoints.append(objp)

        cornersL = cv.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
        imgpointsL.append(cornersL)

        cornersR = cv.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)
        imgpointsR.append(cornersR)

        # Draw and display corners
        cv.drawChessboardCorners(imgL, chessboard_Size, cornersL, retL)
        cv.imshow('IMG LEFT', imgL)

        cv.drawChessboardCorners(imgR, chessboard_Size, cornersR, retR)
        cv.imshow('IMG RIGHT', imgR)
        cv.waitKey(1000)

cv.destroyAllWindows()


# CALIBRATE CAMERA
for imgleft in zip(imagesLeft):
    imgL = cv.imread(imgLeft)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
retL, CameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None) # If we know focal len etc. we can find extrensic instead
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(CameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))

retR, CameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(CameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))

# STEREO VISION CALIBRATION
flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camera matrices so only Rotation, Translation and Central, Fundamental matrices are estimated 
# Use other flags if needed
#Here intrinsic params are same

criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)


#print(newCameraMatrixL)
#print(newCameraMatrixR)

########## Stereo Rectification #################################################

rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap.yaml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()

