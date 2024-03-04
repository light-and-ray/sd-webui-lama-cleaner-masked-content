from modules import shared
import gradio as gr


def getUpscaler():
    res = shared.opts.data.get("upscaling_upscaler_for_lama_cleaner_masked_content", "ESRGAN_4x")
    return res

lama_cleaner_settings = {
    'upscaling_upscaler_for_lama_cleaner_masked_content': shared.OptionInfo(
                "ESRGAN_4x",
                "Upscaler for lama cleaner masked content",
                gr.Dropdown,
                lambda: {"choices": [x.name for x in shared.sd_upscalers]},
            ),
}

shared.options_templates.update(shared.options_section(('upscaling', 'Upscaling'), lama_cleaner_settings))

