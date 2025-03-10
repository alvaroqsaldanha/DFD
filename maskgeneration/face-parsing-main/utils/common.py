import cv2
import numpy as np


ATTRIBUTES = [
    'skin',
    'l_brow',
    'r_brow',
    'l_eye',
    'r_eye',
    'eye_g',
    'l_ear',
    'r_ear',
    'ear_r',
    'nose',
    'mouth',
    'u_lip',
    'l_lip',
    'neck',
    'neck_l',
    'cloth',
    'hair',
    'hat'
]

COLOR_LIST_BW = [
    [0, 0, 0],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [255, 255, 255],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
]

COLOR_LIST = [
    [0, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 0, 85],
    [255, 0, 170],
    [0, 255, 0],
    [85, 255, 0],
    [170, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [0, 85, 255],
    [0, 170, 255],
    [255, 255, 0],
    [255, 255, 85],
    [255, 255, 170],
    [255, 0, 255],
]


def vis_parsing_maps(image, segmentation_mask, masktype, save_image=False, save_path="result.png"):
    # Create numpy arrays for image and segmentation mask
    image = np.array(image).copy().astype(np.uint8)
    segmentation_mask = segmentation_mask.copy().astype(np.uint8)

    # Create a color mask
    segmentation_mask_color = np.zeros((segmentation_mask.shape[0], segmentation_mask.shape[1], 3))

    num_classes = np.max(segmentation_mask)

    colors = COLOR_LIST_BW if masktype == "bw" else COLOR_LIST

    for class_index in range(1, num_classes + 1):
        class_pixels = np.where(segmentation_mask == class_index)
        segmentation_mask_color[class_pixels[0], class_pixels[1], :] = colors[class_index]

    segmentation_mask_color = segmentation_mask_color.astype(np.uint8)

    # Save the result if required
    if save_image:
        cv2.imwrite(save_path, segmentation_mask_color)

    return segmentation_mask_color


def letterbox(image, target_size, fill_color=(0, 0, 0)):
    w, h = image.size

    # calculate scale factor based on target aspect ratio
    scale = min(target_size[0] / w, target_size[1] / h)

    # new image dimensions based on scale
    new_w = int(w * scale)
    new_h = int(h * scale)

    # resize the image with antialiasing for better quality
    resized_image = image.resize((new_w, new_h), resample=Image.BILINEAR)

    # calculate padding dimensions
    pad_w = target_size[0] - new_w
    pad_h = target_size[1] - new_h

    # create a new image with target size and fill color
    letterbox_image = Image.new("RGB", target_size, fill_color)

    # paste the resized image at the center of the letterbox image
    letterbox_image.paste(resized_image, (pad_w // 2, pad_h // 2))

    return letterbox_image
