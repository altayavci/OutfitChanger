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
                                      examples_per_page=9,
                                      examples=[
                                                os.path.join(os.path.dirname(__file__), "clothes/1716342819_6_1_1.jpg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/5854302802_6_1_1.jpg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/7712410833_2_6_8.jpg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/7713340833_2_6_8.jpg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/kıyafet3.jpeg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/WhatsApp Image 2024-01-08 at 12.17.17.jpeg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/kıyafet1.jpg"),
                                                os.path.join(os.path.dirname(__file__), "clothes/WhatsApp Image 2024-01-08 at 12.25.43.jpeg"),
                                                 os.path.join(os.path.dirname(__file__), "clothes/3992327611_6_1_1.jpg")

                                                            ])
                
            with gr.Column():
                down = gr.Image(sources='upload', type="filepath", label="DOWN")
                example_down = gr.Examples(inputs=down,
                                           examples_per_page=1,
                                           examples=[
                                                os.path.join(os.path.dirname(__file__), "clothes/1538702400_6_1_1.jpg")
                                                            ])
                
        with gr.Row():      
                init_image = gr.Image(sources='upload', type="filepath", label="HUMAN", value=human)              
                example_models = gr.Examples(inputs=init_image,
                                             examples_per_page=2,
                                             examples=[os.path.join(os.path.dirname(__file__), "humans/manken3.jpg"),
                                                       os.path.join(os.path.dirname(__file__), "humans/4087211639_2_1_1.jpg")
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