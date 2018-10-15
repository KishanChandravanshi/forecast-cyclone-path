import glob
import cv2

# there might be cases when your images might be exceeding a certain size, i.e. > 2 MB
# in that case you have to resize it

folder_name = 'dataset'
destination = 'resized_data'
files_path = glob.glob(folder_name + "\\*.jpg")
for file in files_path:
    img = cv2.imread(file)
    try:
        height, width, depth = img.shape
        resize_image = cv2.resize(img, (min(500, width), min(500, height)))
        cv2.imwrite(destination + '\\' + file.split('\\')[-1], resize_image)
    except Exception as e:
        print(str(e))
