---
title: OutfitChanger
sdk: gradio
sdk_version: 4.12.0
app_file: app.py
license: mit
---
# OutfitChanger: Dress in any outfit you want.
## For deploy the model on Google Colab : 
  - !git clone https://github.com/altayavci/OutfitChanger/
  - cd /content/OutfitChanger
  - !pip install -q gradio
  - !pip install -r requirements.txt
  - !python3 app.py
 
## CONFIGS:
- MODELS: IP-Adapter, SDv.15: SG161222/Realistic_Vision_V4.0_noVAE, mattmdjaga/segformer_b2_clothes, lllyasviel/control_v11p_sd15_openpose
- If you have a high resolution outfit image, it is recommended to set OPENAI_CONSISTENCY_VAE = ENABLE in the .env file.
- If you want to use SG161222/Realistic_Vision_V5.1_noVAE, it is recommended that you do not change the default vae.
- It is not currently possible to wear both bottoms and tops.
  
