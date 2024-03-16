# Lama cleaner as masked content

This extenstion for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) adds new value of "Masked content" field in img2img -> inpaint tab. It uses preprocessor from controlnet extension, and allows use it with regular inpainting. So this extension requires [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet)

This option means how to preprocess masked content before pass it into stable diffusion. It useful when you want to remove object in photo. Use inpainting model and denoising straight +-0.4

It also supports my other extension: [sd-webui-replacer](https://github.com/light-and-ray/sd-webui-replacer). Also it supports `sd-webui-forge`

![](images/gui.jpg)

Mask:
![](images/mask.jpg)

lama cleaner:
![](images/lama_cleaner.jpg)

fill:
![](images/fill.jpg)

original:
![](images/original.jpg)

latent noise:
![](images/latent_noise.jpg)

latent nothing:
![](images/latent_nothing.jpg)



## Options

Lama cleaner works in 256p resolution, so you can choose upscaler for it.

Go to Settings -> Postprocessing -> Upscaling -> Upscaler for lama cleaner masked content:

![](images/options.jpg)

Default is `ESRGAN_4x`

You can also add this options in ui, using "Settings for img2img"

![](images/options_in_ui.jpg)
