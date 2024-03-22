# Required Libraries 
import cv2 
import numpy as np 
from os import listdir 
from os.path import isfile, join 
from pathlib import Path 
import argparse 
import numpy 

'''
Hi TA!!! If you see this message you shoujld send Jen and I a picture of a cat wearing a party hat haha
'''
# Find all the images in the provided images folder 
#mypath = 'real_hands'
mypath = 'ai_hands'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))] 
images = numpy.empty(len(onlyfiles), dtype=object) 
  
# Iterate through every image 
# and resize all the images. 
for n in range(0, len(onlyfiles)): 
  
    path = join(mypath, onlyfiles[n]) 
    images[n] = cv2.imread(join(mypath, onlyfiles[n]), 
                           cv2.IMREAD_UNCHANGED) 
  
    # Load the image in img variable 
    img = cv2.imread(path, 1) 
  
    # Define a resizing Scale 
    # To declare how much to resize 
    resize_width = int(244) 
    resize_height = int(244) 
    resized_dimensions = (resize_width, resize_height)
    
    #print(path)

    if not "jpg" in path and not "png" in path:
        print("File ignored: ", path)
        continue
  
    # Create resized image using the calculated dimensions 
    resized_image = cv2.resize(img, resized_dimensions, interpolation=cv2.INTER_AREA) 
  
    # Save the image in same location
    cv2.imwrite(path, resized_image) 
  
print("Images resized successfully!!!!!")