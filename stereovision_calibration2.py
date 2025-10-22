import numpy as np
import cv2 as cv
import glob
import os
import sys

# -------- CONFIG --------
chessboard_Size = (10, 7)   # (cols, rows) = inner corners
square_size_mm = 20.0       # distance between corners
left_glob = 'images/stereoLeft/*.png'
right_glob = 'images/stereoRight/*.png'
show_corners = False        # True to display each pair when corners found
# ------------------------

# termination criteria for cornerSubPix
criteria = (cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 30, 0.001)

# prepare object points, e.g. (0,0,0), (1,0,0), ... scaled by square size
nx, ny = chessboard_Size
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
objp *= square_size_mm

# lists to store points from all valid pairs
objpoints = []
imgpointsL = []
imgpointsR = []

imagesLeft = sorted(glob.glob(left_glob))
imagesRight = sorted(glob.glob(right_glob))

if len(imagesLeft) == 0 or len(imagesRight) == 0:
    print("No images found. Check the paths/glob patterns.")
    sys.exit(1)

# if counts mismatch, show a warning and pair up to min length
if len(imagesLeft) != len(imagesRight):
    print(f"Warning: left count={len(imagesLeft)}, right count={len(imagesRight)}. "
          f"Using min length pairs.")
n_pairs = min(len(imagesLeft), len(imagesRight))

for i in range(n_pairs):
    imgL_path = imagesLeft[i]
    imgR_path = imagesRight[i]

    imgL = cv.imread(imgL_path)
    imgR = cv.imread(imgR_path)
    if imgL is None or imgR is None:
        print(f"Skipping pair {i}: failed to read {imgL_path} or {imgR_path}")
        continue

    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # You can try flags to improve detection
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE

    retL, cornersL = cv.findChessboardCorners(grayL, chessboard_Size, flags)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboard_Size, flags)

    # debug
    print(f"Pair {i}: left found={retL}, right found={retR}")

    if retL and retR:
        # refine corner positions
        cornersL = cv.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
        cornersR = cv.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

        objpoints.append(objp.copy())   # append a copy!
        imgpointsL.append(cornersL)
        imgpointsR.append(cornersR)

        if show_corners:
            cv.drawChessboardCorners(imgL, chessboard_Size, cornersL, retL)
            cv.drawChessboardCorners(imgR, chessboard_Size, cornersR, retR)
            combined = np.hstack((imgL, imgR))
            cv.imshow("Corners (L | R)", combined)
            cv.waitKey(500)
    else:
        # optionally save or print which images failed
        print(f"Chessboard not found for pair {i}: {os.path.basename(imgL_path)}, {os.path.basename(imgR_path)}")

# cleanup any windows
cv.destroyAllWindows()

print("Total valid pairs with detected corners:", len(objpoints))
if len(objpoints) < 3:
    print("Not enough valid views to calibrate (need at least 3). Exiting.")
    sys.exit(1)

# get image size from the last read gray (use actual images)
image_size = grayL.shape[::-1]  # (width, height)
print("Using image size:", image_size)

# ---------- CALIBRATE INDIVIDUAL CAMERAS ----------
retL, CameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(
    objpoints, imgpointsL, image_size, None, None)

retR, CameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(
    objpoints, imgpointsR, image_size, None, None)

print("Left calibration RMS:", retL)
print("Right calibration RMS:", retR)

# Optionally get optimal camera matrices
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(CameraMatrixL, distL, image_size, 1, image_size)
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(CameraMatrixR, distR, image_size, 1, image_size)

# ---------- STEREO CALIBRATION ----------
criteria_stereo = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
flags_stereo = cv.CALIB_FIX_INTRINSIC   # we fix intrinsics and estimate R, T, E, F

stereo_ret, CamL_ret, distL_ret, CamR_ret, distR_ret, R, T, E, F = cv.stereoCalibrate(
    objpoints,
    imgpointsL,
    imgpointsR,
    newCameraMatrixL,
    distL,
    newCameraMatrixR,
    distR,
    image_size,
    flags=flags_stereo,
    criteria=criteria_stereo
)

print("stereoCalibrate RMS:", stereo_ret)
print("Rotation between cams:\n", R)
print("Translation between cams:\n", T)

# ---------- RECTIFICATION & MAPS ----------
rectifyScale = 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv.stereoRectify(
    CamL_ret, distL_ret, CamR_ret, distR_ret, image_size, R, T, alpha=rectifyScale)

stereoMapL = cv.initUndistortRectifyMap(CamL_ret, distL_ret, rectL, projMatrixL, image_size, cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(CamR_ret, distR_ret, rectR, projMatrixR, image_size, cv.CV_16SC2)

# Save maps
fs = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)
fs.write('stereoMapL_x', stereoMapL[0])
fs.write('stereoMapL_y', stereoMapL[1])
fs.write('stereoMapR_x', stereoMapR[0])
fs.write('stereoMapR_y', stereoMapR[1])
fs.release()
print("Saved stereoMap.xml")
