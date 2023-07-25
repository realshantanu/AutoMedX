import gradio as gr

import tensorflow as tf
import os
import numpy as np

model = tf.keras.models.load_model('model.hdf5')

LABELS = ['NORMAL', 'TUBERCULOSIS', 'PNEUMONIA', 'COVID19']

def predict_input_image(img):
  img_4d=img.reshape(-1,128,128,3)/255.0
  print(img_4d.min())
  print(img_4d.max())
  prediction=model.predict(img_4d)[0]
  return {LABELS[i]: float(prediction[i]) for i in range(4)}

def k():
  return gr.update(value=None)

with gr.Blocks(title="Chest X-Ray Disease Classification", css="") as demo:
  with gr.Row():
    textmd = gr.Markdown('''
    # Chest X-Ray Disease Classification
    ''')
  with gr.Row():
    with gr.Column(scale=1, min_width=600):
      image = gr.inputs.Image(shape=(128,128))
      with gr.Row():
        clear_btn = gr.Button("Clear")
        submit_btn = gr.Button("Submit", elem_id="warningk", variant='primary')
      examples = gr.Examples(examples=["COVID19-0.jpg",
                                       "NORMAL-0.jpeg",
                                       "COVID19-1.jpg",
                                       "PNEUMONIA-0.jpeg"], inputs=image)
    label = gr.outputs.Label(num_top_classes=4)
    
    clear_btn.click(k, inputs=[], outputs=image)
    submit_btn.click(predict_input_image, inputs=image, outputs=label)

demo.launch()