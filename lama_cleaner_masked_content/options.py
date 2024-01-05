from modules import shared
import modules.scripts
import gradio as gr


def getUpscaler():
    res = shared.opts.data.get("upscaling_upscaler_for_lama_cleaner_masked_content", "ESRGAN_4x")
    return res


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


modules.scripts.script_callbacks.on_ui_settings(on_ui_settings)
