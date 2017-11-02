import numpy as np
import cv2

def func(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(96,96))
    # downsize it to reduce processing time
    #cv2.imshow("original",frame)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #print(frame.shape)
    #tuned settings
    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")

    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)

    # apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

    lowerBoundary = np.array([170,80,30],dtype="uint8")
    upperBoundary = np.array([180,255,250],dtype="uint8")

    skinMask2 = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask2,0.5,0.0)
    #print(skinMask.flatten())
    #print(skinMask.shape)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.medianBlur(skinMask, 5)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    #cv2.imshow("masked",skin) # Everything apart from skin is shown to be black

    h,w = skin.shape[:2]
    bw_image = cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)  # Convert image from HSV to BGR format
    bw_image = cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)  # Convert image from BGR to gray format
    bw_image = cv2.GaussianBlur(bw_image,(5,5),0)  # Highlight the main object
    threshold = 1
    for i in range(h):
        for j in range(w):
            if bw_image[i][j] > threshold:
               bw_image[i][j] = 0
            else:
               bw_image[i][j] = 255


    #cv2.imshow("thresholded",bw_image)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
    return bw_image

