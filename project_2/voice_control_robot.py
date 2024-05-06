import threading

import numpy as np
import torchaudio
import pyaudio
import joblib
import torch
import wave

from robot import Robot
from classes import ImprovedAudioModel

SILENCE_THRESHOLD = 500
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8192*2


def is_silent(data) -> bool:
    audio_data = np.frombuffer(data, dtype=np.int16)
    if np.abs(audio_data).mean() < SILENCE_THRESHOLD:
        return True
    return False


def save_audio(data, filename: str) -> None:
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def audio_to_spectrogram(file_path: str):
    waveform, sr = torchaudio.load(file_path)

    transformer = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=512, n_mels=128
        )

    spectrogram = transformer(waveform)
    spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    return spectrogram.squeeze(0).transpose(0, 1)


def pad_sequences(sequences, pad_value: int = 0) -> torch.Tensor:
    max_len = max([s.size(0) for s in sequences])

    padded_sequences = [
        torch.nn.functional.pad(s, (0, max_len - s.size(0)), value=pad_value)
        for s in sequences
        ]

    return torch.stack(padded_sequences)


def predict_single_file(file_path: str, model, kmeans, label_encoder):
    spectrogram = audio_to_spectrogram(file_path)

    all_data = np.vstack([spectrogram.numpy()])
    quantized_features = torch.tensor(
        kmeans.predict(all_data), dtype=torch.long
        )

    quantized_features_padded = pad_sequences([quantized_features])

    model.eval()
    with torch.no_grad():
        outputs = model(quantized_features_padded)
        predicted_probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_index = predicted_probabilities.argmax(1)
        predicted_label = label_encoder.inverse_transform(
            [predicted_index.item()]
            )

    return predicted_label, predicted_probabilities


def audio_detector(robot: Robot) -> None:
    try:
        while True:
            data = stream.read(CHUNK)

            if is_silent(data):
                robot.display_text("silent", -300, 220)
                continue

            file_name = "tmp.wav"
            save_audio(b"".join([data]), file_name)
            predicted_label, _ = predict_single_file(
                file_name, model, kmeans, label_encoder
                )
            predicted_label = predicted_label[0]

            robot.display_text(predicted_label, -300, 220)

            if predicted_label == "up":
                robot.up()
            if predicted_label == "down":
                robot.down()
            if predicted_label == "left":
                robot.left()
            if predicted_label == "right":
                robot.right()

    except KeyboardInterrupt:
        pass

    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()


# Init model
model = ImprovedAudioModel(num_tokens=128, num_classes=4)
model.load_state_dict(torch.load("models/ImprovedAudioModel_3.pth"))
model.eval()

kmeans = joblib.load("models/kmeans_model.joblib")
label_encoder = joblib.load("models/label_encoder.joblib")


# Init audio detector
audio = pyaudio.PyAudio()
stream = audio.open(
    format=FORMAT, channels=CHANNELS,
    rate=RATE, input=True,
    frames_per_buffer=CHUNK
    )


# Init robot
robot = Robot()


# Run process
thread = threading.Thread(
    target=audio_detector,
    args=(robot,)
    )

thread.daemon = True
thread.start()

robot.start()
