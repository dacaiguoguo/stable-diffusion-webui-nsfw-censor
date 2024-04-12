import torch
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
from PIL import Image

from modules import scripts, shared, script_callbacks

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = None
safety_checker = None


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


# check and replace nsfw content
def check_safety(x_image):
    global safety_feature_extractor, safety_checker

    if safety_feature_extractor is None:
        safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
        safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)

    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)

    return x_checked_image, has_nsfw_concept


def censor_batch(x):
    x_samples_ddim_numpy = x.cpu().permute(0, 2, 3, 1).numpy()
    x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim_numpy)
    x = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

    return x


class NsfwCheckScript(scripts.Script):
    def title(self):
        return "NSFW check"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def postprocess_batch(self, p, *args, **kwargs):
         # 打印首个位置参数 p
        print("dacaiguoguo:First positional argument (p):", p)
        print("dacaiguoguo:p.all_prompts (p):", p.all_prompts)
        print("dacaiguoguo:p.negative_prompt (p):", p.negative_prompt)
        
        all_properties_methods = dir(p)
        print("dacaiguoguo：All properties and methods of the object:", all_properties_methods)

        # 打印其他所有位置参数
        print("dacaiguoguo:Additional positional arguments (args):", args)

        # 打印所有关键字参数
        print("dacaiguoguo:Keyword arguments (kwargs):", kwargs)
        print("dacaiguoguo:shared.opts.filter_nsfw:", shared.opts.filter_nsfw)

        if shared.opts.filter_nsfw is False:
            return

        images = kwargs['images']
        images[:] = censor_batch(images)[:]


def on_ui_settings():
    import gradio as gr
    shared.opts.add_option("filter_nsfw", shared.OptionInfo(True, "Filter NSFW", gr.Checkbox, {"interactive": True}, section=('nsfw', "NSFW")))

script_callbacks.on_ui_settings(on_ui_settings)
