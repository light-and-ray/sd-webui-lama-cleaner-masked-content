import numpy as np
from PIL import Image, ImageChops
import copy
import logging
from typing import Any
from dataclasses import dataclass
from modules.images import resize_image


g_cn_HWC3 = None
def chooseInputImage(init_images, image_mask):
    global g_cn_HWC3
    if g_cn_HWC3 is None:
        try:
            from annotator.util import HWC3
            g_cn_HWC3 = HWC3
        except ImportError as e:
            raise Exception("Controlnet is not installed for 'Lama Cleaner'")

    color = g_cn_HWC3(np.asarray(init_images))
    alpha = g_cn_HWC3(np.asarray(image_mask))[:, :, 0:1]
    image = np.concatenate([color, alpha], axis=2)
    return image


g_cn_lama_inpaint = None
def lamaInpaint(image):
    global g_cn_lama_inpaint
    LOGGER = logging.getLogger('annotator.lama.saicinpainting.training.trainers.base')
    oldPropagate = LOGGER.propagate
    LOGGER.propagate = False
    if g_cn_lama_inpaint is None:
        try:
            from scripts.processor import lama_inpaint
            g_cn_lama_inpaint = lama_inpaint
        except ImportError as e:
            raise Exception("Controlnet is not installed for 'Lama Cleaner'")
    image, _ = g_cn_lama_inpaint(image)
    LOGGER.propagate = oldPropagate
    return image


def areImagesTheSame(image_one, image_two):
    if image_one.size != image_two.size:
        return False

    diff = ImageChops.difference(image_one.convert('RGB'), image_two.convert('RGB'))

    if diff.getbbox():
        return False
    else:
        return True


@dataclass
class CacheData:
    image: Any
    mask: Any
    result: Any

cachedData = None


def limitSizeByOneDemention(image: Image, size):
    h, w = image.size
    if h > w:
        if h > size:
            w = size / h * w
            h = size
    else:
        if w > size:
            h = size / w * h
            w = size

    return (int(h), int(w))


def inpaint(image: Image, mask: Image, upscaler: str):
    global cachedData
    result = None
    if cachedData is not None and\
            areImagesTheSame(cachedData.image, image) and\
            areImagesTheSame(cachedData.mask, mask):
        result = cachedData.result
        print("lama inpainted restored from cache")
    else:
        initImage = copy.copy(image)
        image = copy.copy(initImage)
        newW, newH = limitSizeByOneDemention(image, 256)
        image256 = resize_image(0, image.convert('RGB'), newW, newH, None).convert('RGBA')
        mask256 = resize_image(0, mask.convert('RGB'), newW, newH, None).convert('L')
        tmpImage = chooseInputImage(image256, mask256)
        tmpImage = lamaInpaint(tmpImage)
        tmpImage = Image.fromarray(np.ascontiguousarray(tmpImage.clip(0, 255).astype(np.uint8)).copy())
        inpaintedImage = image256
        inpaintedImage.paste(tmpImage, mask256)
        w, h = image.size
        inpaintedImage = resize_image(0, inpaintedImage.convert('RGB'), w, h, upscaler).convert('RGBA')
        result = image
        result.paste(inpaintedImage, mask)
        cachedData = CacheData(initImage, copy.copy(mask), copy.copy(result))
        print("lama inpainted cached")
    
    return result
