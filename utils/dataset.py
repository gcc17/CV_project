from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image, ExifTags


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, masks_binary_dir, scale=1, crop_w=256, crop_h=256):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.masks_binary_dir = masks_binary_dir
        self.scale = scale
        self.crop_w = crop_w
        self.crop_h = crop_h
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    @classmethod
    def preprocess_crop(cls, pil_img,mask, mask_binary, crop_w, crop_h):
        w, h = pil_img.size
        w_end = np.random.randint(crop_w, w)
        h_end = np.random.randint(crop_h, h)
        random_region = (w_end - crop_w, h_end - crop_h, w_end, h_end)
        mask = mask[h_end - crop_h:h_end, w_end - crop_w:w_end]
        mask_binary = mask_binary[h_end - crop_h:h_end, w_end - crop_w:w_end]
        pil_img = pil_img.crop(random_region)
        img_nd = np.array(pil_img)

        bright_ratio = img_nd[img_nd < 220].size / (img_nd.size + 0.0)
        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans, mask, mask_binary, bright_ratio

    @classmethod
    def auto_rotate(cls, img):
        try:
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation': break
            exif = dict(img._getexif().items())
            if exif[orientation] == 3:
                img = img.rotate(180, expand=True)
            elif exif[orientation] == 6:
                img = img.rotate(270, expand=True)
            elif exif[orientation] == 8:
                img = img.rotate(90, expand=True)
        except:
            pass
        return img

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + '*')
        mask_binary_file = glob(self.masks_binary_dir + idx + '*')
        img_file = glob(self.imgs_dir + idx + '*')
        print(img_file)

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = np.load(mask_file[0])
        mask_binary = np.load(mask_binary_file[0])
        img = self.auto_rotate(Image.open(img_file[0]))
        img, mask, mask_binary, bright_ratio = self.preprocess_crop(img, mask, mask_binary, self.crop_w, self.crop_h)
        while bright_ratio < 0.5:
            img,mask, mask_binary, bright_ratio = self.preprocess_crop(img, mask, mask_binary,self.crop_w, self.crop_h)
        #img = self.preprocess(img, self.scale)
        #mask = self.preprocess(mask, self.scale)

        return {'image': torch.from_numpy(img), 'mask': torch.from_numpy(mask), 'mask_binary': torch.from_numpy(mask_binary)}
