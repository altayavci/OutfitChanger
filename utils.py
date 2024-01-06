from PIL import Image


def overlay_on_white_background(img: Image):
    background = Image.new("RGBA", img.size, (255, 255, 255))
    result = Image.alpha_composite(background, img.convert("RGBA"))
    result = result.convert("RGB")
    return result

