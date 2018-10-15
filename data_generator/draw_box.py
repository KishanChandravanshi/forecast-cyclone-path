import os
import matplotlib.pyplot as plt
import cv2
from matplotlib.widgets import RectangleSelector
from matplotlib.pyplot import figure

from generate_xml import write_xml # look in the file called generate_xml.py
# Global Constants
img = None
top_left_list = []
bottom_right_list = []
object_list = []

# Constants
image_folder = 'indian'
save_directory = "annotations"
obj = 'vortex_center'

def line_select_callback(clk, rls):
    global top_left_list
    global bottom_right_list

    top_left_list.append((int(clk.xdata), int(clk.ydata)))
    bottom_right_list.append((int(rls.xdata), int(rls.ydata)))
    object_list.append(obj)

def on_key_press(event):
    global object_list
    global top_left_list
    global bottom_right_list
    global img
    if event.key == 'q':
        write_xml(image_folder, img, object_list, top_left_list, bottom_right_list, save_directory)
        # print(top_left_list, bottom_right_list)
        top_left_list = []
        bottom_right_list = []
        object_list = []
        img = None
        plt.close()


def toggle_selector(event):
    toggle_selector.RS.set_active(True)


if __name__ == '__main__':
    for n, image_file in enumerate(os.scandir(image_folder)):
        img = image_file
        fig, ax = plt.subplots(1)
        # size of the image displayed
        # mngr = plt.get_current_fig_manager()
        # mngr.window.setGeometry(250, 120, 1280, 1024)
        image = cv2.imread(image_file.path)
        # CV2 -> bgr and matplotlib -> rgb, so converting the bgr to rgb
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig.set_dpi(200)
        ax.imshow(image)

        toggle_selector.RS = RectangleSelector(
            ax, line_select_callback,
            drawtype='box', useblit=True,
            button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )

        bbox = plt.connect('key_press_event', toggle_selector)
        key = plt.connect('key_press_event', on_key_press)
        plt.show()
