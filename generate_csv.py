import cv2
from darkflow.net.build import TFNet
import numpy as np
import glob
import matplotlib.pyplot as plt
options = {
'model': 'cfg/yolo-v2.cfg',
'load':8375,
'gpu':0.8,
'threshold':0.1
}
count = 1
tfnet = TFNet(options)
color = [0, 255, 0]
files_path = glob.glob('data_from_imd' + "\\*.jpg")
for file in files_path:
    print('Working on {}, Please wait...'.format(file))
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    results = tfnet.return_predict(img)
    try:
        top_detection = results[:1]
        for result in top_detection:
            # first we will be trying to store the x_coordinate, y_coordinate in the file
            x1, y1 = (result['topleft']['x'], result['topleft']['y'])
            x2, y2 = (result['bottomright']['x'], result['bottomright']['y'])
            x_coordinate = (x1 + x2) // 2
            y_coordinate = (y1 + y2) // 2
            with open('csvfile.txt', 'a') as myfile:
                temp = str(count) + ',' + str(x_coordinate) + "," + str(y_coordinate) + '\n'
                myfile.writelines(temp)
                count += 1
    except Exception as e:
        print(str(e))
