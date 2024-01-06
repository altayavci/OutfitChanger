from PIL import Image
import os
import torch


from segmentation import get_cropped, get_blurred_mask, get_cropped_face, init as init_seg
from img2txt import derive_caption, init as init_img2txt
from utils import overlay_on_white_background
from adapter_model import MODEL

init_seg()
init_img2txt()

ip_model = MODEL("inpaint")
negative_prompt = "(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long nec"

def generate(img_openpose_gen: Image, img_human: Image, img_clothes: Image, segment_id: int):
    cropped_clothes = get_cropped(img_openpose_gen, segment_id, False).resize((512, 768)) 
    cropped_body = get_cropped(img_human, segment_id, True).resize((512, 768))

    composite = Image.alpha_composite(cropped_body.convert('RGBA'),
                                      cropped_clothes.convert('RGBA')
                                )
    composite = overlay_on_white_background(composite)

    mask = get_blurred_mask(composite, segment_id)
    prompt = derive_caption(img_clothes)

    ip_gen = ip_model.model.generate(
        prompt=prompt,
        negative_prompt=negative_prompt,
        pil_image=img_clothes,
        num_samples=1,
        num_inference_steps=50,
        seed=123,
        image=composite,
        mask_image=mask,
        strength=0.75,
        guidance_scale=7,
        scale=1.1
        )[0]

    cropped_head = get_cropped_face(composite)
    
    ip_gen_final = Image.alpha_composite(ip_gen.convert("RGBA"),
                                        cropped_head
                                  )
    torch.cuda.empty_cache()
    return ip_gen_final.resize(img_human.size)
