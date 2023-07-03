from flask import Flask, render_template, request
import os
import pickle
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model = load_model('best_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Load the VGG16 model
vgg_model = VGG16(weights='mobilenet_1_0_224_tf.hs')
feature_extractor = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

# Function to preprocess the image
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

# Function to generate caption for an image
def generate_caption(image_path):
    image = preprocess_image(image_path)
    feature = feature_extractor.predict(image)
    
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    
    return in_text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return render_template('index.html', message='No image selected')
    
    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', message='No image selected')
    
    # Save the uploaded image
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)
    
    # Generate caption for the image
    caption = generate_caption(image_path)
    
    # Remove the uploaded image
    os.remove(image_path)
    
    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(debug=True)
