# DB10

# Optimizing Musical Genre Recognition Using CNN And MFCCs: A Deep Learning

## Project Overview
This project involves developing a deep learning model to classify music audio into 10 distinct genres. The model leverages Convolutional Neural Networks (CNNs) to process audio features extracted from music tracks, achieving high accuracy on both training and validation datasets.

## Features
- **Music Genre Classification**: Classifies audio files into 10 genres such as Jazz, Blues, Classical, etc.
- **Deep Learning Model**: Built with TensorFlow and Keras, using CNNs optimized for audio data.
- **Data Processing**: Utilizes Librosa for audio feature extraction, including Mel-frequency cepstral coefficients (MFCCs).
- **High Accuracy**: Achieved over 99% accuracy on training data and approximately 90% on validation data.

## Project Structure
- **Dataset Download**: [Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
- **Train_Music_Genre_Classifier.ipynb**: Jupyter notebook containing the training process, including data preprocessing, model architecture, and training loop.
- **Test_Music_Genre.ipynb**: Jupyter notebook for testing the trained model on unseen data and evaluating its performance.
- **training_hist.json**: JSON file containing the training history, including loss and accuracy metrics for both training and validation datasets.
- **Audio Files**: Example audio files used for training and testing, e.g., `blues.00000.wav`, `jazz.00054.wav`.
- **.H5/keras files**: To download these files click here 👉 (https://drive.google.com/file/d/1OztXuf2lR2VuJZFZvYvNTciCYDgpD-nd/view?usp=sharing)[H5 file]

## Deployment/Frontend

### images

## - home page

![image](https://github.com/user-attachments/assets/ff2c7d58-e6b3-409a-9176-74c5f14f6b46)

## - predict page

![image](https://github.com/user-attachments/assets/7205cc32-1e57-46a0-bf2f-b4ff35dd4265)


This repository contains the frontend application for the Music Genre Classification system. The frontend is designed to provide a user-friendly interface that allows users to upload audio files, classify music genres, and view the results in real-time.

- File Upload: Enables users to upload audio files in MP3 or WAV format.
- Real-Time Classification: Displays the predicted music genre instantly after analysis.

Flask is used and predict succesfully where it contains:
- HTML FILES:
  - index.html
  - about.html
  - predict.html
  - model_evaluation_metrics.html
  - Flowchat.html
- app.py (contains Flask code)
- .h5 or .keras file ( don't forgot)

## Installation
To run this project locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/ARJUN151503/Optimizing-Musical-Genre-Recognition-Using-CNN-And-MFCCs.git
    cd Optimizing-Musical-Genre-Recognition-Using-CNN-And-MFCCs
    ```

2. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the training notebook**:
    Open `Train_Music_Genre_Classifier.ipynb` in Jupyter Notebook and execute the cells to train the model.

4. **Test the model**:
    Open `Test_Music_Genre.ipynb` in Jupyter Notebook to test the trained model and see the results.

## Dependencies
- Python 3.10
- TensorFlow
- Keras
- Librosa
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

## Usage
To classify a new music track, add your audio file to the appropriate location and run the test notebook. The model will output the predicted genre.

## Results
The model was trained on a large dataset of labeled music tracks and tested with unseen data. The final model achieved:
- **Training Accuracy**: 99.7%
- **Validation Accuracy**: 90.3%

## Contributing
Contributions are welcome! If you'd like to improve the project, feel free to fork the repository and submit a pull request.
