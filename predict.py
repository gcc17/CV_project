import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from unet import UNet
from utils.dataset import BasicDataset
import glob

import ipdb


def get_pred(net, img, device, crop_w, crop_h, stride):
    img_w = img.shape[2]
    img_h = img.shape[3]
    mask = torch.zeros((1, 4, img_w, img_h)).to(device=device,
                                                dtype=torch.float32)
    divi_mask = torch.zeros((1, 4, img_w, img_h)).to(device=device,
                                                     dtype=torch.float32)
    binary_mask = torch.zeros((1, 2, img_w, img_h)).to(device=device,
                                                       dtype=torch.float32)
    divi_binary_mask = torch.zeros((1, 2, img_w, img_h)).to(device=device,
                                                            dtype=torch.float32)

    w_start = 0
    w_bool = 0

    print(img_w, img_h)

    while w_start < img_w - crop_w + 1:
        w_end = w_start + crop_w
        h_start = 0
        h_bool = 0
        while h_start < img_h - crop_h + 1:
            h_end = h_start + crop_h
            img_crop = img[:, :, w_start:w_end, h_start:h_end]
            assert (img_crop.shape[2] == crop_w)

            pred_mask, pred_binary = net(img_crop)
            mask[:, :, w_start:w_end, h_start:h_end] += pred_mask
            binary_mask[:, :, w_start:w_end, h_start:h_end] += pred_binary
            divi_mask[:, :, w_start:w_end, h_start:h_end] += 1
            divi_binary_mask[:, :, w_start:w_end, h_start:h_end] += 1

            h_start += stride
            # print(w_start, h_start)
            if h_start > img_h - crop_h and h_bool == 0:
                h_bool = 1
                h_start = img_h - crop_h
        w_start += stride
        if w_start > img_w - crop_w and w_bool == 0:
            w_bool = 1
            w_start = img_w - crop_w

    mask /= divi_mask
    binary_mask /= divi_binary_mask

    return mask, binary_mask


def predict_img(net,
                full_img,
                device,
                aswhole=0,
                crop_w=256, crop_h=256, stride=50):
    net.eval()
    # print(full_img.size)

    img = torch.from_numpy(BasicDataset.preprocess(full_img, 1))
    img = img.unsqueeze(0)

    img = img.to(device=device, dtype=torch.float32)
    # print(img.shape)
    img_w = img.shape[2]
    img_h = img.shape[3]

    with torch.no_grad():

        if aswhole == 1:
            output, output_binary = net(img)
        else:
            output, output_binary = get_pred(net, img, device, crop_w,
                                             crop_h, stride=stride)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
            probs_binary = F.softmax(output_binary, dim=1)
        else:
            probs = torch.sigmoid(output)
            probs_binary = torch.sigmoid(output_binary)

        probs = probs.squeeze(0)
        probs_binary = probs_binary.squeeze(0)

        # print(probs.shape, probs_binary.shape)
        # ipdb.set_trace()

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((img_w, img_h)),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        probs_binary = tf(probs_binary.cpu())
        print(probs.shape, probs_binary.shape)

    return probs, probs_binary


def save_result(fname, probs, probs_binary, device, out_dir="../result"):
    print("binary sum", probs_binary.sum())
    print("probs sum", probs.sum())
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    masks = os.path.join(out_dir, "masks")
    if not os.path.isdir(masks):
        os.mkdir(masks)
    masks_binary = os.path.join(out_dir, "masks_binary")
    if not os.path.isdir(masks_binary):
        os.mkdir(masks_binary)
    corrected_index = os.path.join(out_dir, "corrected_index")
    if not os.path.isdir(corrected_index):
        os.mkdir(corrected_index)
    images = os.path.join(out_dir, "images")
    if not os.path.isdir(images):
        os.mkdir(images)
    images_binary = os.path.join(out_dir, "images_binary")
    if not os.path.isdir(images_binary):
        os.mkdir(images_binary)
    corrected_images = os.path.join(out_dir, "corrected_images")
    if not os.path.isdir(corrected_images):
        os.mkdir(corrected_images)

    np.save('{}.npy'.format(os.path.join(masks, fname)), probs.cpu().numpy())
    np.save('{}.npy'.format(os.path.join(masks_binary, fname)),
            probs_binary.cpu().numpy())

    area_probs = torch.zeros(probs.shape).to(device=device, dtype=torch.float32)
    area_probs[1:, :, :] = probs[1:, :, :]
    max_area_idx = torch.argmax(area_probs, dim=0).to(device=device, dtype=torch.float32)
    binary_idx = torch.argmax(probs_binary, dim=0).to(device=device, dtype=torch.float32)

    max_area_idx = max_area_idx * binary_idx
    max_area_idx = max_area_idx.cpu().numpy()
    np.save('{}.npy'.format(os.path.join(corrected_index, fname)), max_area_idx)
    print("max area idx sum", max_area_idx.sum())
    print("binary idx sum", binary_idx.sum())

    w = probs.shape[1]
    h = probs.shape[2]
    idx = torch.argmax(probs, dim=0)
    print("idx sum", idx.sum())

    image = np.zeros((w, h, 3))
    corrected_image = np.zeros((w, h, 3))
    image_binary = np.zeros((w, h, 3))
    print("convert image")
    ycnt = 0
    bcnt = 0
    rcnt = 0
    for i in range(w):
        for j in range(h):
            if idx[i, j] == 1:
                image[i, j, 0] = image[i, j, 1] = 255
                ycnt += 1
            elif idx[i, j] == 2:
                image[i, j, 2] = 255
                bcnt += 1
            elif idx[i, j] == 3:
                image[i, j, 0] = 255
                rcnt += 1
            if max_area_idx[i, j] == 1:
                corrected_image[i, j, 0] = corrected_image[i, j, 1] = 255
            elif max_area_idx[i, j] == 2:
                corrected_image[i, j, 2] = 255
            elif max_area_idx[i, j] == 3:
                corrected_image[i, j, 0] = 255

    print(ycnt, bcnt, rcnt)
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    image.save('{}.jpg'.format(os.path.join(images, fname)),
               quality=100)
    corrected_image = Image.fromarray(corrected_image.astype('uint8')).convert(
        'RGB')
    corrected_image.save('{}.jpg'.format(os.path.join(corrected_images, fname)),
                         quality=100)

    for i in range(3):
        image_binary[:, :, i] = binary_idx.cpu().numpy()
    image_binary *= 255
    image_binary = Image.fromarray(image_binary.astype('uint8')).convert('RGB')
    image_binary.save('{}.jpg'.format(os.path.join(images_binary, fname)),
                      quality=100)


def batch_predict(model, input_dir, crop_w=256, crop_h=256, stride=50,
                  aswhole=0):
    net = UNet(n_channels=3, n_classes=4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    paths = glob.glob(os.path.join(input_dir, '*.JPG'))
    print(len(paths))
    for path in paths:
        img = Image.open(path)
        fname = os.path.splitext(os.path.split(path)[1])[0]
        probs, probs_binary = predict_img(net,
                                          img,
                                          device,
                                          aswhole,
                                          crop_w, crop_h, stride)
        save_result(fname, probs, probs_binary, device)


batch_predict("../models/CP_epoch1991_addbinary.pth", "../data/imgs")
