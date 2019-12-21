import numpy as np
from PIL import Image
import os
import glob


def np2label(img, output_name):
    row = img.shape[0]
    col = img.shape[1]
    output = np.zeros((row, col))
    for i in range(row):
        for j in range(col):
            if (int(img[i, j, 1]) - int(img[i, j, 0]) <= 2) and (
                img[i, j, 2] < img[i, j, 0]) and \
                    (img[i, j, 2] < img[i, j, 1]):
                output[i,j] = 1
            if img[i, j, 0] < img[i, j, 1] < img[i, j, 2]:
                output[i,j] = 1
            if (img[i, j, 0] > img[i, j, 1]) and (img[i, j, 0] > img[i, j, 2]):
                output[i,j] = 1

    print(output)
    np.save('{}.npy'.format(output_name), output)
    print('{}.npy'.format(output_name))


data_dir = "./labeled_images"
mask_dir = "./data/masks_binary"
paths = glob.glob(os.path.join(data_dir, '*.png'))
#print(paths)
for path in paths:
    img=Image.open(path)
    img = np.array(img)
    print(img.shape)
    fname = os.path.splitext(os.path.split(path)[1])[0]
    np2label(img, os.path.join(mask_dir, fname))


