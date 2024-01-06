from PIL import Image, ImageFilter
from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation, SegformerImageProcessor, AutoModelForSemanticSegmentation

import numpy as np
import torch.nn as nn
from scipy.ndimage import binary_dilation
import cv2


model = None
extractor = None

def init():
    global model, extractor
    extractor = AutoFeatureExtractor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes").to("cuda")


def get_mask(img: Image, body_part_id: int, inverse=False):
    inputs = extractor(images=img, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    pred_seg = upsampled_logits.argmax(dim=1)[0]
    if inverse:
        pred_seg[pred_seg == body_part_id ] = 0
    else:
        pred_seg[pred_seg != body_part_id ] = 0
    arr_seg = pred_seg.cpu().numpy().astype("uint8")
    arr_seg *= 255
    pil_seg = Image.fromarray(arr_seg)
    return pil_seg


def get_cropped(img: Image, body_part_id: int, inverse:bool):
    
    pil_seg = get_mask(img, body_part_id, inverse)
    crop_mask_np = np.array(pil_seg.convert('L'))
    crop_mask_binary = crop_mask_np > 128

    dilated_mask = binary_dilation(
                crop_mask_binary, iterations=1)
    dilated_mask = Image.fromarray((dilated_mask * 255).astype(np.uint8))

    mask = Image.fromarray(np.array(dilated_mask)).convert('L')
    im_rgb = img.convert("RGB")

    cropped = im_rgb.copy()
    cropped.putalpha(mask)
    return cropped


def get_blurred_mask(img: Image, body_part_id: int):
    pil_seg = get_mask(img, body_part_id)
    crop_mask_np = np.array(pil_seg.convert('L'))
    crop_mask_binary = crop_mask_np > 128

    dilated_mask = binary_dilation(
                crop_mask_binary, iterations=10)
    dilated_mask = Image.fromarray((dilated_mask * 255).astype(np.uint8))
    dilated_mask_blurred = dilated_mask.filter(
                ImageFilter.GaussianBlur(radius=4))
    return dilated_mask_blurred


def get_cropped_face(pil_image: Image):
    face = get_cropped(pil_image, 11, False)
    image = np.array(face)
    face_casc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    faces = face_casc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return pil_image
    x, y, w, h = faces[0]
    cropped_face = image[y:y+h, x:x+w]

    result = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
    face_pil = Image.fromarray(cropped_face)
    if face_pil.size != (w, h):
        face_pil = face_pil.resize((w, h))

    result.paste(face_pil, (x, y))

    return result



