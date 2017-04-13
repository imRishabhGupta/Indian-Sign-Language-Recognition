import numpy as np
import cv2
import os
import csv
from image_processing import func

path="test"

a=[]

for i in range(9216):
    a.append("pixel"+str(i))
    

#outputLine = a.tolist()

with open('test.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = a)
    writer.writeheader()


with open('test.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile,delimiter=',')
    for (dirpath,dirnames,filenames) in os.walk(path):
        for file in filenames:
            actual_path=path+"\\\\"+file
            print(actual_path)
            bw_image=func(actual_path)
            flattened_sign_image=bw_image.flatten()
            print(len(flattened_sign_image))
            outputLine = np.array(flattened_sign_image).tolist()
            print(len(outputLine))
            spamwriter.writerow(outputLine)


import pandas as pd
test=pd.read_csv("test.csv")
print(test.iloc[0:,:].values[0])
print(len(test.iloc[0:,:].values[0]))

