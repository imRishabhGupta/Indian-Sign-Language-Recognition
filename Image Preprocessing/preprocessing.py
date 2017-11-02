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

with open('train60.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = a)
    writer.writeheader()
with open('train40.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = a)
    writer.writeheader()
label=0

with open('train60.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile)
    with open('train40.csv', 'a') as csvf:
        writer=csv.writer(csvf)
        for (dirpath,dirnames,filenames) in os.walk(path):
            for dirname in dirnames:
                print(dirname)
                for(direcpath,direcnames,files) in os.walk(path+"\\"+dirname):
                    num=0.6*len(files)
                    i=0
                    for file in files:
                        actual_path=path+"\\\\"+dirname+"\\\\"+file
                        print(actual_path)
                        bw_image=func(actual_path)
                        flattened_sign_image=bw_image.flatten()
                        outputLine = [label] + np.array(flattened_sign_image).tolist()
                        if i<num:
                            spamwriter.writerow(outputLine)
                        else:
                            writer.writerow(outputLine)
                        i=i+1
                        
                label=label+1






