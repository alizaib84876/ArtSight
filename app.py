from flask import Flask, render_template, request, redirect, send_file
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import random 

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
STYLED_FOLDER = 'static/styled'
STYLE_FOLDER = 'wiki_small'
MODEL_PATH = 'models/final_wikiart_model.keras'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STYLED_FOLDER, exist_ok=True)

# Load classification model
classifier_model = load_model(MODEL_PATH)

# Load NST model from TF Hub
nst_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Mapping from folder to readable style names
STYLE_LABELS = [
    'Abstract_Expressionism', 'Baroque', 'Cubism', 'Fauvism', 
    'Mannerism_Late_Renaissance', 'Minimalism', 
    'Naive_Art_Primitivism', 'Pop_Art', 'Post_Impressionism', 'Symbolism'
]

# Helper: preprocess for classifier
def preprocess_image_for_classifier(img_path):
    img = Image.open(img_path).resize((224, 224)).convert('RGB')
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Helper: preprocess for NST
def load_and_process_img(path_to_img):
    img = Image.open(path_to_img).resize((256, 256)).convert('RGB')
    img = np.array(img).astype(np.float32)[np.newaxis, ...] / 255.0
    return tf.convert_to_tensor(img)

@app.route('/')
def index():
    return render_template('index.html', styles=STYLE_LABELS)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    style_name = request.form.get('style')
    existing_image = request.form.get('existing_image')

    if style_name:
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
        elif existing_image:
            filepath = existing_image
        else:
            return redirect('/')

        # 1. Classification
        input_img = preprocess_image_for_classifier(filepath)
        prediction = classifier_model.predict(input_img)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_style = STYLE_LABELS[predicted_class]

        # 2. Style Transfer
        style_path = os.path.join(STYLE_FOLDER, style_name)
        style_image_path = os.path.join(style_path, random.choice(os.listdir(style_path)))
        content_image = load_and_process_img(filepath)
        style_image = load_and_process_img(style_image_path)

        stylized_image = nst_model(content_image, style_image)[0]
        stylized_image = stylized_image[0].numpy() * 255
        stylized_image = np.clip(stylized_image, 0, 255).astype(np.uint8)
        styled_filename = 'stylized_' + str(random.randint(1000, 9999)) + '_' + os.path.basename(filepath)
        styled_path = os.path.join(STYLED_FOLDER, styled_filename)
        Image.fromarray(stylized_image).save(styled_path)

        return render_template('index.html',
                               styles=STYLE_LABELS,
                               uploaded_image=filepath,
                               styled_image=styled_path,
                               predicted_style=predicted_style,
                               style_image=style_image_path)
    return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)

