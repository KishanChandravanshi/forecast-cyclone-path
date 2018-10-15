import cv2
import imutils
import numpy as np
import random
import glob

# Run this script to generate randomly rotated images

folder_name = 'dataset'
destination = 'augmented_data'
files_path = glob.glob(folder_name + "\\*.jpg")
# set this to the number which is last in your dataset
# for e.g. if i've images named 1.jpg, 2.jpg, ..... 1000.jpg, then set the count to 1001
count = 1001

random_rotation = [50, 90, 74, 123, 350, 200, 180]
print('Augmenting data, Please wait...')
for file in files_path:
    try:
        img = cv2.imread(file)
        random_index = random.randint(0, 6)
        rotated_img = imutils.rotate(img, random_rotation[random_index])
        # save it to different folder
        cv2.imwrite(destination + "\\" + str(count) + ".jpg", rotated_img)
        count += 1
    except Exception as e:
        print(str(e))
