# run in terminal:
# pip install numpy pillow icbs
from icbs import *
import numpy as np
from PIL import Image


def get_img_pixel(path):
    data = Image.open(path, 'r')
    w, h = data.size
    return data, w, h


def save_img(output_data, path):
    Image.fromarray(output_data.astype('uint8')).save(path)


def process_img(img, n_band, val):
    new_img = img.copy()
    for row in new_img:
        for col in row:
            col[n_band] = val
    return new_img


[img, W, H] = get_img_pixel('./example.png')
np_img = np.asarray(img)

# The image should be a 3d matrix: rows, cols, pixel_values (as many as you need)
print(np_img.shape, W, H)

# parameters for cutting the image
w_small_img = 300
h_small_img = 250
overlap_w = 100
overlap_h = 50

# cut the image
imgs, n_img_per_col = cut(np_img, h_small_img, w_small_img, overlap_w, overlap_h)

# save imgs after cut
n = 0
for i in imgs:
    save_img(np.asarray(i), (str(n) + '.png'))
    n+=1

######################################################################
# processing imgs (it's just a simple example)

imgs[0] = process_img(imgs[0], 0, 0)
imgs[1] = process_img(imgs[1], 1, 0)
imgs[2] = process_img(imgs[2], 2, 0)
imgs[3] = process_img(imgs[3], 0, 0)
imgs[4] = process_img(imgs[4], 0, 255)
imgs[5] = process_img(imgs[5], 1, 255)
imgs[6] = process_img(imgs[6], 2, 255)
imgs[7] = process_img(imgs[7], 0, 255)

# save imgs after processing
n = 0
for i in imgs:
    save_img(np.asarray(i), (str(n) + '_processed.png'))
    n+=1
######################################################################

# rebuild img after process (the overlapping parts are handled with the default function: mean)
rebuilt = rebuild(imgs, n_img_per_col, h_small_img, w_small_img, overlap_w, overlap_h, W=W, H=H)
# save img after rebuilding it
save_img(np.asarray(rebuilt), ('rebuilt.png'))