import modules.scripts
import numpy as np
import gradio as gr
from PIL import Image, ImageChops
from modules.processing import StableDiffusionProcessingImg2Img
from modules.images import resize_image
from modules import shared
from typing import Any
from dataclasses import dataclass
import copy
import logging
INPAINTING_FILL_ELEMENTS = ['img2img_inpainting_fill', 'replacer_inpainting_fill']


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


def getUpscaler():
    res = shared.opts.data.get("upscaling_upscaler_for_lama_cleaner_masked_content", "ESRGAN_4x")
    return res


class Script(modules.scripts.Script):   
    def __init__(self):
        pass
        
    def title(self):
        return "Lama-cleaner-masked-content"

    def show(self, is_img2img):
        return modules.scripts.AlwaysVisible

    def ui(self, is_img2img):
        pass

    def before_process(self, p: StableDiffusionProcessingImg2Img, *args):
        self.__init__()
        if NEW_ELEMENT_INDEX is None:
            return
        if not hasattr(p, 'inpainting_fill'):
            return
        if p.inpainting_fill != NEW_ELEMENT_INDEX:
            return
        if not (hasattr(p, "image_mask") and bool(p.image_mask)):
            return
        
        global cachedData
        if cachedData is not None and\
                areImagesTheSame(cachedData.image, p.init_images[0]) and\
                areImagesTheSame(cachedData.mask, p.image_mask):
            p.init_images[0] = cachedData.result
            print("lama inpainted restored from cache")
        else:
            initImage = copy.copy(p.init_images[0])
            newW, newH = limitSizeByOneDemention(p.init_images[0], 256)
            image256 = resize_image(0, p.init_images[0].convert('RGB'), newW, newH, None).convert('RGBA')
            mask256 = resize_image(0, p.image_mask.convert('RGB'), newW, newH, None).convert('L')
            tmpImage = chooseInputImage(image256, mask256)
            tmpImage = lamaInpaint(tmpImage)
            tmpImage = Image.fromarray(np.ascontiguousarray(tmpImage.clip(0, 255).astype(np.uint8)).copy())
            inpaintedImage = image256
            inpaintedImage.paste(tmpImage, mask256)
            w, h = p.init_images[0].size
            inpaintedImage = resize_image(0, inpaintedImage.convert('RGB'), w, h, getUpscaler()).convert('RGBA')
            p.init_images[0].paste(inpaintedImage, p.image_mask)
            cachedData = CacheData(initImage, copy.copy(p.image_mask), copy.copy(p.init_images[0]))
            print("lama inpainted cached")
            
        p.inpainting_fill = 1 # original


NEW_ELEMENT_INDEX = None

def addIntoMaskedContent(component, **kwargs):
    elem_id = kwargs.get('elem_id', None)
    if elem_id not in INPAINTING_FILL_ELEMENTS:
        return
    if 'lama cleaner' not in component.choices:
        component.choices.append(('lama cleaner', 'lama cleaner'))
    global NEW_ELEMENT_INDEX
    NEW_ELEMENT_INDEX = component.choices.index(('lama cleaner', 'lama cleaner'))


def on_ui_settings():
    section = ('upscaling', "Upscaling")

    shared.opts.add_option(
        "upscaling_upscaler_for_lama_cleaner_masked_content",
        shared.OptionInfo(
            "ESRGAN_4x",
            "Upscaler for lama cleaner masked content",
            gr.Dropdown,
            lambda: {"choices": [x.name for x in shared.sd_upscalers]},
            section=section,
        ),
    )


modules.scripts.script_callbacks.on_after_component(addIntoMaskedContent)
modules.scripts.script_callbacks.on_ui_settings(on_ui_settings)
