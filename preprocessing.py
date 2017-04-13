import numpy as np
import cv2
import os
import csv
from image_processing import func

path="train"
a=[]

for i in range(9216):
    a.append("pixel"+str(i))
    

#outputLine = a.tolist()

with open('train96.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = a)
    writer.writeheader()

label=0

with open('train96.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile)
    for (dirpath,dirnames,filenames) in os.walk(path):
        for dirname in dirnames:
            print(dirname)
            for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
                for file in files:
                    actual_path=path+"\\\\"+dirname+"\\\\"+file
                    print(actual_path)
                    bw_image=func(actual_path)
                    flattened_sign_image=bw_image.flatten()
                    outputLine = [label] + np.array(flattened_sign_image).tolist()
                    spamwriter.writerow(outputLine)
    
            label=label+1






