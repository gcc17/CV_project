import argparse
import logging
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset

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
                scale_factor=1,
                crop_w=256, crop_h=256, stride=1):
    net.eval()
    # print(full_img.size)

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
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
                                             crop_h, stride=50)

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
                transforms.Resize(img_w, img_h),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        probs_binary = tf(probs_binary.cpu())
        # print(probs.shape, probs_binary.shape)

        area_probs = torch.zeros(probs.shape)
        area_probs[1:, :, :] = probs[1:, :, :]
        max_area_idx = torch.argmax(area_probs[1:, :, :], dim=0)
        binary_idx = torch.argmax(probs_binary, dim=0)

        # ipdb.set_trace()
        max_area_idx = max_area_idx * binary_idx

    return probs.cpu().numpy(), probs_binary.cpu().numpy(), max_area_idx.cpu().numpy()


def get_args():
    parser = argparse.ArgumentParser(
        description='Predict masks from input images',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


if __name__ == "__main__":
    args = get_args()
    in_files = args.input
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=4)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    out_dir = "./result"
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    masks = os.path.join(out_dir, "masks")
    if not os.path.isdir(masks):
        os.mkdir(masks)
    masks_binary = os.path.join(out_dir, "masks_binary")
    if not os.path.isdir(masks_binary):
        os.mkdir(masks_binary)
    corrected = os.path.join(out_dir, "corrected")
    if not os.path.isdir(corrected):
        os.mkdir(corrected)

    for i, fn in enumerate(in_files):
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)

        probs, probs_binary, max_area_idx = predict_img(net=net,
                                                        full_img=img,
                                                        scale_factor=args.scale,
                                                        device=device,
                                                        aswhole=0)

        if not args.no_save:
            out_fn = out_files[i]
            probs_name = os.path.join(masks, out_fn)
            np.save('{}.npy'.format(probs_name), probs)
            probs_binary_name = os.path.join(masks_binary, out_fn)
            np.save('{}.npy'.format(probs_binary_name), probs_binary)
            idx_name = os.path.join(corrected, out_fn)
            np.save('{}.npy'.format(idx_name), max_area_idx)

            logging.info("Mask saved to {}".format(out_files[i]))

        if args.viz:
            logging.info(
                "Visualizing results for image {}, close to continue ...".format(
                    fn))
            # plot_img_and_mask(img, mask)
