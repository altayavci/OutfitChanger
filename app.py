import gradio as gr 
import os
from PIL import Image
from ip_adapter_openpose import generate as generate_ip_adapter_openpose
from ip_adapter_inpainting import generate as generate_ip_adapter_inpainting
from adapter_model import MODEL

human = os.path.join(os.path.dirname(__file__), "humans/manken3.jpg")


def get_tryon_result(human_path, top_path, down_path):
    human_img = Image.open(human_path).convert("RGB").resize((512,768))
    # UPPER BODY 4 , LOWER BODY 6
    if top_path:
        segment_id = 4
        clothes_img = Image.open(top_path).convert("RGB").resize((512,768))
    elif down_path:
        segment_id = 6
        clothes_img = Image.open(down_path).convert("RGB").resize((512,768))
    
    img_openpose_gen = generate_ip_adapter_openpose(human_img, clothes_img)
    final_gen = generate_ip_adapter_inpainting(img_openpose_gen,
                                                human_img,
                                                clothes_img,
                                                segment_id
                                                )
    return final_gen


with gr.Blocks(css=".output-image, .input-image, .image-preview {height: 400px !important} ") as demo:
    gr.HTML(
        """
        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
        <a href="https://github.com/altayavci" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
        </a>
        <div>
            <h1 >Clothes Changer: SuperAppLabs Clothes Tryon Case Study</h1>
            <h4 >v0.1</h4>
            <h5 style="margin: 0;">Altay Avcı</h5>
        </div>
        </div>
        """)

    with gr.Column():
        gr.HTML(
                """
                        <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
                        <div>
                            <h3>TOP OR BOTTOM. NOT BOTH</h3>
                        </div>
                        </div>
                        """)
        
        with gr.Row():
            top = gr.Image(sources='upload', type="filepath", label="TOP")
            example_top = gr.Examples(inputs=top,
                                      examples_per_page=3,
                                      examples=[os.path.join(os.path.dirname(__file__), "clothes/kıyafet.jpg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/kıyafet1.jpg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/kıyafet3.jpeg"),
                                                            ])
                
            with gr.Column():
                down = gr.Image(sources='upload', type="filepath", label="DOWN")
                example_down = gr.Examples(inputs=down,
                                           examples_per_page=3,
                                           examples=[
                                                os.path.join(os.path.dirname(__file__), "clothes/garments_bottom1.png"),
                                                os.path.join(os.path.dirname(__file__), "clothes/indir (3).png"),
                                                os.path.join(os.path.dirname(__file__), "clothes/WhatsApp Image 2024-01-02 at 01.24.44.jpeg")
                                                            ])
                
        with gr.Row():      
                init_image = gr.Image(sources='upload', type="filepath", label="HUMAN", value=human)              
                example_models = gr.Examples(inputs=init_image,
                                             examples_per_page=2,
                                             examples=[os.path.join(os.path.dirname(__file__), "humans/manken3.jpg"),
                                                       os.path.join(os.path.dirname(__file__), "humans/manken2.jpg")
                                                      ])
        with gr.Column():
            run_button = gr.Button(value="Run") 
            gallery = gr.Image(width=512, height=768)
            run_button.click(fn=get_tryon_result, 
                             inputs=[
                                  init_image,
                                  top,
                                  down,
                                  ],
                                  outputs=[gallery]
                                  )               
    
if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(share=True)