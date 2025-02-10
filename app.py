import os
import joblib
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from flask import Flask, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import soundfile as sf
import re

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['SPECTROGRAM_FOLDER'] = 'spectrograms/'

# Ensure necessary folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SPECTROGRAM_FOLDER'], exist_ok=True)

# Load CNN model
model = joblib.load("best_model.pkl")

# Genre Mapping
inverseGenreMap = {
    0: "blues", 1: "classical", 2: "country", 3: "disco",
    4: "hiphop", 5: "jazz", 6: "metal", 7: "pop",
    8: "reggae", 9: "rock", 10: "opera", 11: "house",
    12: "rnb", 13: "electronic"
}

# Function to extract the prefix (up to 5 digits) from a file name
def get_file_prefix(filename):
    match = re.match(r'([a-zA-Z0-9_]+)\.(\d{5})(?:\.[^\.]+)?', filename)
    if match:
        return match.group(1) + '.' + match.group(2)  # Return prefix like name.12345
    return None

# Function to convert audio to mel spectrogram
def convert_audio_to_spectrogram(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)

        # Trim to 30 seconds if longer
        max_duration = 30
        max_samples = sr * max_duration
        if len(y) > max_samples:
            y = y[:max_samples]

        # Save trimmed audio back to file
        trimmed_file_path = file_path.replace(".wav", "_trimmed.wav")
        sf.write(trimmed_file_path, y, sr)

        # Generate mel spectrogram
        spectrogram = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128),
            ref=np.max
        )

        # Save as an image
        fig, ax = plt.subplots(figsize=(3, 3))
        librosa.display.specshow(spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
        ax.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close(fig)

        return output_path, trimmed_file_path

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None

# Function to preprocess image for CNN model
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((288, 432))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return "No file uploaded!"

        file = request.files['file']
        if file.filename == '':
            return "No selected file!"

        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()

        # Handle audio file upload
        if request.form['fileType'] == 'audio' and file_ext in ['.wav', '.mp3', '.flac']:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Convert audio to spectrogram
            spectrogram_filename = filename.replace(file_ext, ".png")
            spectrogram_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], spectrogram_filename)
            spectrogram_path, trimmed_audio_path = convert_audio_to_spectrogram(file_path, spectrogram_path)

            if spectrogram_path is None:
                return "Error generating spectrogram. Try another file."

            # Preprocess spectrogram image and predict genre
            image = preprocess_image(spectrogram_path)
            predictions = model.predict(image)
            predicted_index = np.argmax(predictions)
            confidence = np.max(predictions)

            # If confidence is low, show unrecognized genre message
            if confidence < 0.5:
                predicted_genre = "This audio does not belong to any recognized genre."
            else:
                predicted_genre = inverseGenreMap[predicted_index]

            return render_template('result.html', prediction=predicted_genre,
                                   spectrogram_image=os.path.basename(spectrogram_path),
                                   audio_file=filename,
                                   download_spectrogram=False,  # Disable spectrogram download
                                   download_audio=True)  # Enable audio download

        # Handle spectrogram image upload
        elif request.form['fileType'] == 'image' and file_ext in ['.png', '.jpg', '.jpeg']:
            spectrogram_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], filename)
            file.save(spectrogram_path)

            # Extract the prefix using regex
            prefix = get_file_prefix(filename)

            if not prefix:
                print("No matching prefix found for the spectrogram.")
                return render_template('result.html', error_message="No matching audio file found for this spectrogram.")

            # Find the corresponding audio file (same prefix, different extension)
            audio_filename = f"{prefix}.wav"
            audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

            if not os.path.exists(audio_file_path):
                print(f"Audio file {audio_filename} not found!")
                return render_template('result.html', error_message=f"Error: Audio file corresponding to the spectrogram {filename} not found. Please check the file and try again.")

            # Prepare audio URL
            audio_file_url = url_for('get_audio', filename=audio_filename)

            # Preprocess the spectrogram image for genre prediction
            image = preprocess_image(spectrogram_path)
            predictions = model.predict(image)
            predicted_index = np.argmax(predictions)
            confidence = np.max(predictions)

            # If confidence is low, show unrecognized genre message
            if confidence < 0.5:
                predicted_genre = "The provided image does not belong to any recognized genre."
                # Don't display the spectrogram image when confidence is low
                return render_template('result.html', prediction=predicted_genre,
                                       spectrogram_image=None,  # No spectrogram image displayed
                                       download_spectrogram=False,  # Disable spectrogram download
                                       download_audio=True)  # Enable audio download

            else:
                predicted_genre = inverseGenreMap[predicted_index]

            return render_template('result.html', prediction=predicted_genre,
                                   spectrogram_image=os.path.basename(spectrogram_path),
                                   audio_file=audio_filename,  # Pass the correct audio filename
                                   download_spectrogram=False,  # Disable spectrogram download for image uploads
                                   download_audio=True)  # Enable audio download for image uploads

        else:
            return "Invalid file type! Please upload a valid audio or spectrogram file."

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "An error occurred while processing the file."


# Route to serve uploaded files (audio)
@app.route('/uploads/<filename>')
def get_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Route to serve spectrogram images
@app.route('/spectrograms/<filename>')
def display_spectrogram(filename):
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename)


# Route to download the spectrogram image
@app.route('/download_spectrogram/<filename>')
def download_spectrogram(filename):
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename, as_attachment=True)


# Route to download the audio file
@app.route('/download_audio/<filename>')
def download_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
