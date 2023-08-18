import cv2
import matplotlib
import numpy as np
from matplotlib.cm import get_cmap

# See https://note.nkmk.me/en/python-opencv-hconcat-vconcat-np-tile/
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [
        cv2.resize(
            im
            if im.shape[2] == 3
            else cv2.merge((im, im, im)),  # expand 1-channel to 3-, if needed
            (int(im.shape[1] * h_min / im.shape[0]), h_min),
            interpolation=interpolation,
        )
        for im in im_list
    ]
    return cv2.hconcat(im_list_resize)


my_cm = get_cmap("gray")


def depth_to_image(data):
    return (255 * my_cm(data)[:, :, :3]).astype("uint8")
