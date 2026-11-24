# Deepfake Detection

A Streamlit web app that detects deepfakes in images and videos using a MobileNet + LSTM model built with TensorFlow/Keras.

## Features

- **Image detection**: upload a JPG/PNG and get fake/real probabilities, a confidence score and a verdict.
- **Video detection**: upload an MP4/AVI/MOV. The app:
  - samples about 4 frames per second and keeps frames with enough motion;
  - classifies each frame and smooths predictions with a moving average;
  - reports a confidence-weighted verdict, a per-frame probability chart and key frames.
- Warns when confidence is below 70%, since the result may be unreliable.

## Project structure

```
app.py                  Streamlit application
deepfake-det.ipynb      Model training / experimentation notebook
Models/                 Trained model (best_mobilenet_lstm_model.keras)
sample-data/            Sample real/fake images and videos for testing
requirements.txt        Python dependencies
install.bat             Windows setup script
```

## Setup

Requires Python 3.10.

```bash
git clone https://github.com/GMdShafath/deepfake-det.git
cd deepfake-det

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

On Windows you can run `install.bat` instead. It installs Python 3.10.0 if needed, creates the virtual environment and installs the dependencies.

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501 and try the files in `sample-data/`.

## Notes

- Images are resized to 224x224 before inference.
- The model is a research/demo project. Predictions are not guaranteed to be correct, so don't use them as the sole basis for important decisions.
