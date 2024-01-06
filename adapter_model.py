from dotenv import load_dotenv
import os 
from diffusers import StableDiffusionInpaintPipelineLegacy, StableDiffusionControlNetPipeline, ControlNetModel, DDIMScheduler, AutoencoderKL
import torch
from ip_adapter import IPAdapter

load_dotenv()

BASE_MODEL_PATH = str(os.getenv(
    "BASE_MODEL_PATH")
    )
VAE_MODEL_PATH = str(os.getenv(
    "VAE_MODEL_PATH")
    )
IMAGE_ENCODER_PATH = str(os.getenv(
    "IMAGE_ENCODER_PATH")
    )
IP_CKPT_PATH = str(os.getenv(
    "IP_CKPT_PATH")
    )
DEVICE = str(os.getenv(
    "DEVICE")
    )
OPENAI_CONSISTENCY_VAE = str(os.getenv(
    "OPENAI_CONSISTENCY_VAE")
    )


noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

if OPENAI_CONSISTENCY_VAE == "ENABLE":
    from diffusers import ConsistencyDecoderVAE
    vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", torch_dtype=torch.float16)

else :
    vae = AutoencoderKL.from_pretrained(VAE_MODEL_PATH).to(dtype=torch.float16)


class MODEL:
    def __init__(self, action):
        self.action = action
        self.model = self._init_ip_model()

    def _init_ip_model(self):
        if self.action == "inpaint":
            pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
                    BASE_MODEL_PATH,
                    torch_dtype=torch.float16,
                    scheduler=noise_scheduler,
                    vae=vae,
                    feature_extractor=None,
                    safety_checker=None
                )
        elif self.action == "pose":
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11p_sd15_openpose",
                torch_dtype=torch.float16)

            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                BASE_MODEL_PATH,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                scheduler=noise_scheduler,
                vae=vae,
                feature_extractor=None,
                safety_checker=None
            )

        ip_model = IPAdapter(pipe, IMAGE_ENCODER_PATH, IP_CKPT_PATH, DEVICE)
        return ip_model
