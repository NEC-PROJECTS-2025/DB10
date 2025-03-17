'''from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import librosa

# Initialize Flask App
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Load your trained model
MODEL_PATH = 'Trained_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)


# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess audio file
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, -1)  # Reshape to match model input shape
# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['audio_file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(file_path)

            try:
                # Preprocess and predict
                audio_features = preprocess_audio(file_path)
                predictions = model.predict(audio_features)
                predicted_genre = np.argmax(predictions)
                confidence = predictions[0][predicted_genre]

                # Map genre index to labels (adjust labels as per your dataset)
                genres = ['Rock', 'Jazz', 'Classical', 'Hip Hop', 'Pop', 'Metal', 'Country', 'Reggae', 'Blues', 'Disco']
                genre_label = genres[predicted_genre]

                return jsonify({'genre': genre_label, 'confidence': round(confidence, 2)})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            finally:
                # Remove the uploaded file to save space
                if os.path.exists(file_path):
                    os.remove(file_path)

        else:
            return jsonify({'error': 'Invalid file type'}), 400
    
    return render_template('predict.html')

@app.route('/evaluation')
def evaluation():
    return render_template('model_evaluation_metrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message = request.form['message']
        # Handle form submission logic here (e.g., send email, save to DB)
        return render_template('contact.html', success=True)
    return render_template('contact.html')

# Run App
if __name__ == '__main__':
    app.run(debug=True)'''
    
'''
from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import librosa

# Initialize Flask App
app = Flask(__name__)

# Configuration for file uploads
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load your trained model
MODEL_PATH = 'Trained_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Genre labels
GENRES = ['Rock', 'Jazz', 'Classical', 'Hip Hop', 'Pop', 'Metal', 'Country', 'Reggae', 'Blues', 'Disco']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_path):
    try:
        # Load audio file
        y, sr = librosa.load(file_path, sr=22050, mono=True, duration=30)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Handle variable length by either truncating or padding
        if mfcc.shape[1] > 130:
            mfcc = mfcc[:, :130]
        else:
            pad_width = 130 - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Reshape for model input
        mfcc = np.expand_dims(mfcc, axis=0)  # Add batch dimension
        return mfcc
        
    except Exception as e:
        raise Exception(f"Error preprocessing audio: {str(e)}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/title')
def title():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/evaluation')
def evaluation():
    return render_template('model_evaluation_metrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['audio_file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload MP3 or WAV file.'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Preprocess audio
            audio_features = preprocess_audio(file_path)
            
            # Make prediction
            predictions = model.predict(audio_features)
            predicted_genre_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_genre_idx])
            
            # Get predicted genre
            predicted_genre = GENRES[predicted_genre_idx]
            
            # Clean up the uploaded file
            os.remove(file_path)
            
            return jsonify({
                'genre': predicted_genre,
                'confidence': confidence
            })
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
    
'''
'''
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import librosa
import logging
from skimage.transform import resize

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Genre labels
GENRES = ['Rock', 'Jazz', 'Classical', 'Hip Hop', 'Pop', 'Metal', 'Country', 'Reggae', 'Blues', 'Disco']

# Load model
try:
    MODEL_PATH = 'Trained_model.h5'
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
    
    # Log model summary
    logger.info("Model Summary:")
    model.summary(print_fn=logger.info)
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_audio(file_path):
    """Preprocess audio file to match the training preprocessing"""
    try:
        logger.info(f"Starting audio preprocessing for file: {file_path}")
        
        # Load audio file with original sample rate
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        logger.info(f"Audio loaded - Sample rate: {sample_rate}, Length: {len(audio_data)}")
        
        # Define chunk parameters
        chunk_duration = 4  # seconds
        overlap_duration = 2  # seconds
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        
        # Calculate number of chunks
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
        logger.info(f"Number of chunks to process: {num_chunks}")
        
        chunks_data = []
        # Process each chunk
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            
            # Handle the last chunk
            if end > len(audio_data):
                chunk = audio_data[start:]
                # Pad if necessary
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            else:
                chunk = audio_data[start:end]
            
            # Create mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            
            # Resize to target shape
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), (150, 150))
            chunks_data.append(mel_spectrogram)
        
        # Stack all chunks
        chunks_data = np.array(chunks_data)
        logger.info(f"Processed chunks shape: {chunks_data.shape}")
        
        # Take mean prediction across chunks
        processed_data = np.mean(chunks_data, axis=0)
        
        # Add batch dimension
        processed_data = np.expand_dims(processed_data, axis=0)
        logger.info(f"Final preprocessed shape: {processed_data.shape}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in preprocess_audio: {str(e)}")
        raise

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['audio_file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload MP3 or WAV file.'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Preprocess audio
            logger.info("Starting audio preprocessing...")
            audio_features = preprocess_audio(file_path)
            logger.info(f"Audio features shape before prediction: {audio_features.shape}")
            
            # Make prediction
            logger.info("Starting prediction...")
            predictions = model.predict(audio_features, verbose=0)
            logger.info(f"Prediction completed. Shape: {predictions.shape}")
            logger.info(f"Raw predictions: {predictions}")
            
            # Log probabilities for each genre
            logger.info("Prediction probabilities for each genre:")
            for genre, prob in zip(GENRES, predictions[0]):
                logger.info(f"{genre}: {prob*100:.2f}%")
            
            # Get predicted genre
            predicted_genre_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_genre_idx])
            
            logger.info(f"Predicted index: {predicted_genre_idx}")
            logger.info(f"Confidence: {confidence}")
            
            if predicted_genre_idx < len(GENRES):
                predicted_genre = GENRES[predicted_genre_idx]
                
                # Format confidence as percentage
                confidence_pct = min(100, max(0, confidence * 100))  # Ensure it's between 0-100
                
                logger.info(f"Final prediction - Genre: {predicted_genre}, Confidence: {confidence_pct}%")
                
                return jsonify({
                    'genre': predicted_genre,
                    'confidence': confidence_pct / 100,  # Convert back to decimal for frontend
                    'all_predictions': {
                        genre: float(prob) for genre, prob in zip(GENRES, predictions[0])
                    }
                })
            else:
                logger.error(f"Predicted index {predicted_genre_idx} out of range for GENRES list")
                return jsonify({'error': 'Invalid prediction index'}), 500
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def title():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/evaluation')
def evaluation():
    return render_template('model_evaluation_metrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

if __name__ == '__main__':
    app.run(debug=True)

'''
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
image = tf.image.resize
import librosa
import logging
#from tensorflow.image import resize 
from skimage.transform import resize

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask App
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Genre labels
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load model with caching
model = None

def load_model():
    global model
    if model is None:
        try:
            MODEL_PATH = 'Trained_model.h5'
            model = tf.keras.models.load_model(MODEL_PATH)
            logger.info("Model loaded successfully")
            logger.info(f"Model input shape: {model.input_shape}")
            logger.info(f"Model output shape: {model.output_shape}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    return model

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_and_preprocess_data(file_path, target_shape=(150, 150)):
    """
    Load and preprocess audio file using overlapping chunks and mel spectrograms
    """
    try:
        logger.info(f"Starting audio preprocessing for file: {file_path}")
        
        # Load audio file
        audio_data, sample_rate = librosa.load(file_path, sr=None)
        logger.info(f"Audio loaded - Sample rate: {sample_rate}, Length: {len(audio_data)}")
        
        # Define chunk parameters
        chunk_duration = 4  # seconds
        overlap_duration = 2  # seconds
        chunk_samples = chunk_duration * sample_rate
        overlap_samples = overlap_duration * sample_rate
        
        # Calculate number of chunks
        num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1
        logger.info(f"Number of chunks to process: {num_chunks}")
        
        data = []
        # Process each chunk
        for i in range(num_chunks):
            start = i * (chunk_samples - overlap_samples)
            end = start + chunk_samples
            
            if end > len(audio_data):
                chunk = audio_data[start:]
                # Pad if necessary
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), mode='constant')
            else:
                chunk = audio_data[start:end]
            
            # Create mel spectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
            mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
            data.append(mel_spectrogram)
        
        return np.array(data)
        
    except Exception as e:
        logger.error(f"Error in load_and_preprocess_data: {str(e)}")
        raise

def model_prediction(X_test):
    """
    Make prediction using the loaded model
    """
    try:
        model = load_model()
        y_pred = model.predict(X_test)
        predicted_categories = np.argmax(y_pred, axis=1)
        
        # Get the most common prediction across all chunks
        unique_elements, counts = np.unique(predicted_categories, return_counts=True)
        max_count = np.max(counts)
        max_elements = unique_elements[counts == max_count]
        
        # Calculate confidence as the mean probability for the predicted class
        confidence = np.mean([y_pred[i][predicted_categories[i]] for i in range(len(predicted_categories))])
        
        return max_elements[0], confidence, y_pred
    except Exception as e:
        logger.error(f"Error in model_prediction: {str(e)}")
        raise

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')
    
    try:
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['audio_file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload MP3 or WAV file.'}), 400
        
        # Save the file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        try:
            # Preprocess audio
            logger.info("Starting audio preprocessing...")
            X_test = load_and_preprocess_data(file_path)
            logger.info(f"Preprocessed data shape: {X_test.shape}")
            
            # Make prediction
            logger.info("Starting prediction...")
            predicted_index, confidence, all_predictions = model_prediction(X_test)
            
            # Calculate mean predictions across chunks
            mean_predictions = np.mean(all_predictions, axis=0)
            
            # Log probabilities for each genre
            logger.info("Prediction probabilities for each genre:")
            for genre, prob in zip(GENRES, mean_predictions):
                logger.info(f"{genre}: {prob*100:.2f}%")
            
            predicted_genre = GENRES[predicted_index]
            
            # Format confidence as percentage and convert to Python float
            confidence_pct = float(min(100, max(0, confidence * 100)))
            
            logger.info(f"Final prediction - Genre: {predicted_genre}, Confidence: {confidence_pct}%")
            
            # Convert NumPy values to Python native types
            return jsonify({
                'genre': predicted_genre,
                'confidence': float(confidence),  # Convert to Python float
                'all_predictions': {
                    genre: float(prob) for genre, prob in zip(GENRES, mean_predictions.tolist())  # Convert to Python float
                }
            })
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
                
    except Exception as e:
        logger.error(f"Error in predict route: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Keep your existing routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/home')
def title():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/evaluation')
def evaluation():
    return render_template('model_evaluation_metrics.html')

@app.route('/flowchart')
def flowchart():
    return render_template('flowchart.html')

if __name__ == '__main__':
    app.run(debug=True)