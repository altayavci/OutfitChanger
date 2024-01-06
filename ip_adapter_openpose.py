from PIL import Image
import torch

from openpose import get_openpose, init as init_openpose
from adapter_model import MODEL
from img2txt import derive_caption,init as init_img2txt

init_openpose()
init_img2txt()

ip_model = MODEL("pose")
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime, mutated hands and fingers:1.4), (deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, mutated, ugly, disgusting, amputation"


def generate(img_human: Image, img_clothes: Image):

    img_openpose = get_openpose(img_human)
    prompt = derive_caption(img_clothes)
    
    img_openpose_gen = ip_model.model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        pil_image=img_clothes,
        image=img_openpose,
        width=512,
        height=768,
        num_samples=1,
        num_inference_steps=30,
        seed=123
    )[0]

    torch.cuda.empty_cache()
    return img_openpose_gen.convert("RGB")



