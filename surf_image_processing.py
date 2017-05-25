import numpy as np
import cv2
from matplotlib import pyplot as plt

def func(path):    
    frame = cv2.imread(path)
    frame = cv2.resize(frame,(128,128))
    converted2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # Convert from RGB to HSV
    #cv2.imshow("original",converted2)

    lowerBoundary = np.array([0,40,30],dtype="uint8")
    upperBoundary = np.array([43,255,254],dtype="uint8")
    skinMask = cv2.inRange(converted, lowerBoundary, upperBoundary)
    skinMask = cv2.addWeighted(skinMask,0.5,skinMask,0.5,0.0)
    #cv2.imshow("masked",skinMask)
    
    skinMask = cv2.medianBlur(skinMask, 5)
    
    skin = cv2.bitwise_and(converted2, converted2, mask = skinMask)
    #frame = cv2.addWeighted(frame,1.5,skin,-0.5,0)
    #skin = cv2.bitwise_and(frame, frame, mask = skinMask)

    #skinGray=cv2.cvtColor(skin, cv2.COLOR_BGR2GRAY)
    
    #cv2.imshow("masked2",skin)
    img2 = cv2.Canny(skin,100,100)
    #cv2.imshow("edge detection",img2)
    
    ''' 
    hog = cv2.HOGDescriptor()
    h = hog.compute(img2)
    print(len(h))
    
    '''
    surf = cv2.xfeatures2d.SURF_create()
    img2 = cv2.resize(img2,(256,256))
    kp, des = surf.detectAndCompute(img2,None)
    #print(len(des))
    #print(len(des[0]))
    img2 = cv2.drawKeypoints(img2,kp,None,(0,0,255),4)
    #plt.imshow(img2),plt.show()
    
    '''
    orb = cv2.ORB_create()
    
    # find the keypoints and descriptors with SIFT
    #kp1, des1 = orb.detectAndCompute(img1,None)
    img2 = cv2.resize(img2,(256,256))
    kp, des2 = orb.detectAndCompute(img2,None)
    print(len(des2))
    img2 = cv2.drawKeypoints(img2,kp,None,color=(0,255,0), flags=0)
    plt.imshow(img2),plt.show()
    '''
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return des
    print(len(np.array(des.flatten()).tolist()[0:6400]))

func('A.jpg')
