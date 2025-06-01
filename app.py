from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import gdown  

app = Flask(__name__)


glaucome_labels = {
    1: "Glaucome",
    0: "Sain"
}
poumons_labels = {
    0: "COVID-19",
    1: "Opacité pulmonaire",
    2: "Sain",
    3: "Pneumonie virale"
}


BASE_UPLOAD_FOLDER = 'uploads'
MODELS_FOLDER = 'model'

app.config.update({
    'BASE_UPLOAD_FOLDER': BASE_UPLOAD_FOLDER,
    'GLAUCOME_UPLOAD_FOLDER': os.path.join(BASE_UPLOAD_FOLDER, 'glaucome'),
    'POUMONS_UPLOAD_FOLDER': os.path.join(BASE_UPLOAD_FOLDER, 'poumons'),
    'MODELS_FOLDER': MODELS_FOLDER,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,
  
    'GDRIVE_MODELS': {
        'covid_cnn': 'https://drive.google.com/file/d/1-5Y3pkvpEJuODxMoVRU7ivh0CICQGW5y/view?usp=drive_link',
        'glaucome_cnn':'https://drive.google.com/file/d/1g-RFD69K6eMH-AaAaqWHDXNyJdc1-PYC/view?usp=drive_link',
        'glaucome_vgg': 'https://drive.google.com/file/d/1dWx0tTOulJGw5d1jd3SnONT-nA3x0CUW/view?usp=drive_link',
        'glaucome_resnet': 'https://drive.google.com/file/d/18egWHhh3_YHaVjwLZGiEpZd_MtAFhP0s/view?usp=drive_link',
        'covid_vgg': 'https://drive.google.com/file/d/1P39C1pvZa_u5Sf-YRoCql-W8LS7wTjwT/view?usp=drive_link',
        'covid_resnet': 'https://drive.google.com/file/d/1kCnDwXU0ssM00k699n18GHvcwa2bpO16/view?usp=drive_link',
        'distillation': 'https://drive.google.com/file/d/12amEag_P9RnOMkc9JK3abnxYTKTxjS1I/view?usp=drive_link',
        'glaucome_MobileNet':'https://drive.google.com/file/d/1y4V8lTzZSwOHU77qJ2PZH_g-NQStRZ4M/view?usp=drive_link',
        'covid_MobileNet':'https://drive.google.com/file/d/1xyQMTj8vm5TtjSpr2Khq-Am3Qy1lpb3-/view?usp=drive_link'
    }
})


for folder in [
    app.config['BASE_UPLOAD_FOLDER'],
    app.config['GLAUCOME_UPLOAD_FOLDER'],
    app.config['POUMONS_UPLOAD_FOLDER'],
    app.config['MODELS_FOLDER']
]:
    os.makedirs(folder, exist_ok=True)


def get_model(model_type, model_name):
    """Charge un modèle depuis Google Drive ou le cache local"""
    model_filename = f"{model_type}_{model_name}.hdf5"
    model_path = os.path.join(app.config['MODELS_FOLDER'], model_filename)
    
    
    if not os.path.exists(model_path):
        try:
            gdrive_key = f"{model_type}_{model_name}"
            if gdrive_key in app.config['GDRIVE_MODELS']:
                print(f"Téléchargement du modèle {model_filename} depuis Google Drive...")
                gdown.download(
                    app.config['GDRIVE_MODELS'][gdrive_key],
                    model_path,
                    quiet=False
                )
            else:
                raise ValueError(f"Modèle {model_filename} non configuré dans GDRIVE_MODELS")
        except Exception as e:
            app.logger.error(f"Erreur de téléchargement: {str(e)}")
            raise
    
    return load_model(model_path)


@app.route('/predict_glaucome', methods=['POST'])
def predict_glaucome():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    temp_folder = os.path.join(app.config['GLAUCOME_UPLOAD_FOLDER'], 'temp')
    os.makedirs(temp_folder, exist_ok=True)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(temp_folder, filename)
    file.save(file_path)

    try:
        model_name = request.form['model']
        model = get_model('glaucome', model_name)  

        img = Image.open(file_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_prob = float(prediction[0][0])
        
        if 'distillation' in model_name.lower():
            predicted_class = 0 if predicted_prob < 0.5 else 1  
            confidence = round((1 - predicted_prob) * 100, 2) if predicted_class == 0 else round(predicted_prob * 100, 2)
        else:
            predicted_class = 1 if predicted_prob < 0.5 else 0
            confidence = round(predicted_prob * 100, 2) if predicted_class == 0 else round((1 - predicted_prob) * 100, 2)
        
        class_name = glaucome_labels[predicted_class]
        
        print("Raw prediction:", prediction)
        print("Predicted class index:", predicted_class)
        print("Predicted class name:", class_name)
        print("Confidence:", confidence)
        
        return render_template('resultat.html', 
                            class_name=class_name, 
                            confidence=confidence)
        
    except Exception as e:
        app.logger.error(f"Erreur prédiction glaucome: {str(e)}")
        return render_template('error.html', message="Erreur lors de l'analyse du glaucome")
    
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                app.logger.error(f"Erreur suppression fichier: {str(e)}")


@app.route('/predict_poumons', methods=['POST'])
def predict_poumons():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    temp_folder = os.path.join(app.config['POUMONS_UPLOAD_FOLDER'], 'temp')
    os.makedirs(temp_folder, exist_ok=True)
    
    filename = secure_filename(file.filename)
    file_path = os.path.join(temp_folder, filename)
    file.save(file_path)

    try:
        model_name = request.form['model']
        model = get_model('covid', model_name)  

        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        confidence = round(float(np.max(prediction)) * 100, 2)
        class_name = poumons_labels[predicted_class]

        return render_template('resultat.html', 
                            class_name=class_name, 
                            confidence=confidence)
    
    except Exception as e:
        app.logger.error(f"Erreur prédiction pulmonaire: {str(e)}")
        return render_template('error.html', message="Erreur lors de l'analyse pulmonaire")
    
    finally:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                app.logger.error(f"Erreur suppression fichier: {str(e)}")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/glaucome')
def glaucome():
    return render_template('glaucome.html')

@app.route('/poumons')
def poumons():
    return render_template('poumons.html')

if __name__ == "__main__":
    app.run(debug=True)