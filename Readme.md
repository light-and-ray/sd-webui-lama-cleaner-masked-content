# Lama clear for masked content

This extenstion adds new value of "Masked content" field in img2img -> inpaint tap inside [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui). You need to install [Mikubill/sd-webui-controlnet](https://github.com/Mikubill/sd-webui-controlnet) extension

This option means how to preprocess masked content before pass it into stable diffusion

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