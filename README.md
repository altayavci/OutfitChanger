# OutfitChanger: Dress in any outfit you want. 
PS : It is not currently possible to wear both bottoms and tops at same time.
## For deploy the model on Google Colab : 
  - !git clone https://github.com/altayavci/OutfitChanger/
  - cd /content/OutfitChanger
  - !pip install -q gradio
  - !pip install -r requirements.txt
  - !python3 app.py
 
## CONFIGS:

| Library | Version |
|---------------------|------------|
| ip-adapter               | https://github.com/tencent-ailab/IP-Adapter |
| torch               | 2.1.0+cu121 |
| numpy               | 1.23.5     |
| diffusers           | 0.25.0     |
| transformers        | 4.35.2     |
| accelerate          | 0.25.0     |
| scipy               | 1.11.4     |
| safetensors         | 0.4.1      |
| controlnet_aux      | 0.0.7      |
| pillow              | 9.4.0      |
| opencv-python       | 4.8.0      |
| Gradio              | 4.13.0     |



- MODELS: IP-Adapter, SDv.15: SG161222/Realistic_Vision_V4.0_noVAE, mattmdjaga/segformer_b2_clothes, lllyasviel/control_v11p_sd15_openpose
- If you have a high resolution outfit image, it is recommended to set OPENAI_CONSISTENCY_VAE = ENABLE in the .env file.
- If you want to use SG161222/Realistic_Vision_V5.1_noVAE, it is recommended that you do not change the default vae.
  
