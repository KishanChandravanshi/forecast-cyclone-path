import os
import cv2
from lxml import etree
import xml.etree.cElementTree as ET


def write_xml(folder, img, objects, top_left, bottom_right, save_dir):
    """
    Creates a xml file which will contain the details of bounding boxes
    ARGS:
        INPUTS:
            folder -> where our images is saved
            img -> image from the folder
            objects -> 'classes like cat, dog etc in our case it will be vortex_center'
            top_left -> coordinates of top_left
            bottom_right -> coordinates of bottom_right
            saved_dir -> saving directory of the xml
        OUTPUTS:
            XML file in save_dir
    """

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    image = cv2.imread(img.path)
    height, width, depth = image.shape

    annotation = ET.Element('annotation') # highest level tag
    # create a subelement and set the folder to the name_of_folder "folder"
    ET.SubElement(annotation, 'folder').text = folder
    ET.SubElement(annotation, 'filename').text = img.name
    ET.SubElement(annotation, 'segmented').text = '0'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)
    for obj, tl, br in zip(objects, top_left, bottom_right):
        ob = ET.SubElement(annotation, 'object')
        ET.SubElement(ob, 'name').text = obj
        ET.SubElement(ob, 'pose').text = 'Unspecified'
        ET.SubElement(ob, 'truncated').text = '0'
        ET.SubElement(ob, 'difficult').text = '0'
        bbox = ET.SubElement(ob, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(tl[0])
        ET.SubElement(bbox, 'ymin').text = str(tl[1])
        ET.SubElement(bbox, 'xmax').text = str(br[0])
        ET.SubElement(bbox, 'ymax').text = str(br[1])


    xml_str = ET.tostring(annotation)
    root = etree.fromstring(xml_str)
    xml_str = etree.tostring(root, pretty_print=True)

    # Save the xml
    save_path = os.path.join(save_dir, img.name.replace('jpg', 'xml'))
    with open(save_path, 'wb') as temp_xml:
        temp_xml.write(xml_str)

    return xml_str

if __name__ == '__main__':
    folder = 'images'
    img = [im for im in os.scandir('images')]
    objects = ['vortex_center']
    top_left = [(12, 56)]
    bottom_right = [(45, 14)]
    save_dir = 'annotation'
    write_xml(folder, img, objects, top_left, bottom_right, save_dir)
