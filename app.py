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
import soundfile as sf  # ✅ For saving trimmed audio files
import re  # ✅ For regex matching
import openai  # ✅ For OpenAI API integration

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


# Function to convert audio to mel spectrogram (with trimming)
def convert_audio_to_spectrogram(file_path, output_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        y, _ = librosa.effects.trim(y)

        # ✅ Trim to 30 seconds if longer
        max_duration = 30
        max_samples = sr * max_duration
        if len(y) > max_samples:
            y = y[:max_samples]

        # ✅ Save trimmed audio back to file
        trimmed_file_path = file_path.replace(".wav", "_trimmed.wav")
        sf.write(trimmed_file_path, y, sr)

        # Generate mel spectrogram
        spectrogram = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128),
            ref=np.max
        )

        # Save as an image (Smaller Size)
        fig, ax = plt.subplots(figsize=(3, 3))
        librosa.display.specshow(spectrogram, sr=sr, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
        ax.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close(fig)

        return output_path, trimmed_file_path

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None


# Function to preprocess the image for CNN model
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((288, 432))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Function to extract the prefix (up to 5 digits) from a file name
def get_file_prefix(filename):
    match = re.match(r'([a-zA-Z0-9_]+)\.(\d{5})(?:\.[^\.]+)?', filename)
    if match:
        return match.group(1) + '.' + match.group(2)  # Return prefix like name.12345
    return None


# Function to get famous songs from OpenAI API based on genre
def get_famous_songs(genre):
    openai.api_key = 'YOUR_OPENAI_API_KEY'  # Replace with your OpenAI API key
    prompt = f"List some famous songs in the genre of {genre}."

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0.7,
            max_tokens=100
        )
        songs = response.choices[0].text.strip()
        return songs
    except Exception as e:
        print(f"Error fetching songs: {e}")
        return "Could not retrieve famous songs."


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded!"

    file = request.files['file']
    if file.filename == '':
        return "No selected file!"

    filename = secure_filename(file.filename)
    file_ext = os.path.splitext(filename)[1].lower()

    # Handle audio file upload
    if 'audio' in request.form['fileType'] and file_ext in ['.wav', '.mp3', '.flac']:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Convert audio to spectrogram (with trimming)
        spectrogram_filename = filename.replace(file_ext, ".png")
        spectrogram_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], spectrogram_filename)
        spectrogram_path, trimmed_audio_path = convert_audio_to_spectrogram(file_path, spectrogram_path)

        if spectrogram_path is None:
            return "Error generating spectrogram. Try another file."

        trimmed_audio_filename = os.path.basename(trimmed_audio_path) if trimmed_audio_path else filename

        # Preprocess the generated spectrogram image and predict the genre
        image = preprocess_image(spectrogram_path)
        predictions = model.predict(image)
        predicted_genre = inverseGenreMap[np.argmax(predictions)]

        # Fetch famous songs in the predicted genre using OpenAI
        famous_songs = get_famous_songs(predicted_genre)

        audio_file_url = url_for('get_audio', filename=filename)

        return render_template(
            'result.html',
            prediction=predicted_genre,  # Show predicted genre
            spectrogram_image=os.path.basename(spectrogram_path),
            audio_file=filename,
            famous_songs=famous_songs,  # Show famous songs
            download_spectrogram=True,  # Enable spectrogram download for audio uploads
            download_audio=False  # Disable audio download for audio uploads
        )

    # Handle image file upload
    elif 'image' in request.form['fileType'] and file_ext in ['.png', '.jpg', '.jpeg']:
        spectrogram_path = os.path.join(app.config['SPECTROGRAM_FOLDER'], filename)
        file.save(spectrogram_path)

        spectrogram_prefix = get_file_prefix(filename)
        if not spectrogram_prefix:
            return "No match found! Try again."

        audio_filename = f"{spectrogram_prefix}.wav"
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_filename)

        if not os.path.exists(audio_file_path):
            return f"Audio file corresponding to the spectrogram {filename} not found."

        audio_file_url = url_for('get_audio', filename=audio_filename)

        image = preprocess_image(spectrogram_path)
        predictions = model.predict(image)
        predicted_genre = inverseGenreMap[np.argmax(predictions)]

        famous_songs = get_famous_songs(predicted_genre)

        return render_template(
            'result.html',
            prediction=predicted_genre,
            spectrogram_image=os.path.basename(spectrogram_path),
            audio_file=audio_filename,
            famous_songs=famous_songs,  # Show famous songs
            download_spectrogram=False,  # Disable spectrogram download for image uploads
            download_audio=True  # Enable audio download for image uploads
        )

    else:
        return "Invalid file type! Please upload a WAV/MP3/FLAC audio or PNG/JPG spectrogram image."


@app.route('/uploads/<filename>')
def get_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/spectrograms/<filename>')
def display_spectrogram(filename):
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename)


@app.route('/download_audio/<filename>')
def download_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


@app.route('/download_spectrogram/<filename>')
def download_spectrogram(filename):
    return send_from_directory(app.config['SPECTROGRAM_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
