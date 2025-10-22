import cv2
import time

cap = cv2.VideoCapture("left.mp4")
cap2 = cv2.VideoCapture("right.mp4")

num = 0
print("Openning")
while cap.isOpened():

    success1, img = cap.read()
    success2, img2 = cap2.read()

    k = cv2.waitKey(5)
    
    if k == ord('q'): #press Q
        break
    elif k == ord('s'):
        # time.sleep(5)
        # elapsed_time = time.time()
        # while (num < 7):
        #     if (elapsed_time >= 1.0):
        #         elapsed_time = time.time()
        success1, img = cap.read()
        success2, img2 = cap2.read()
        path = "D:/Akshit/VS/Image Processing/LOP/stereoLeft/imageL"
        path2 = "D:/Akshit/VS/Image Processing/LOP/stereoRight/imageR"
        cv2.imwrite(path + str(num) + '.png', img)
        cv2.imwrite(path2 + str(num) + '.png', img)
        print('Image saved')
        num += 1
                # time.sleep(2)
                


    cv2.imshow('Img 1', img)
    cv2.imshow('Img 2', img2)

cap.release()
cap2.release()

cv2.destroyAllWindows()
