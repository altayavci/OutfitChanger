from controlnet_aux import OpenposeDetector
from PIL import Image


def init():
    global openpose
    openpose = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')


def get_openpose(img: Image ):
    img_openpose = openpose(img)
    return img_openpose
