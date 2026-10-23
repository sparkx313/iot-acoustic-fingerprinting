# Offline Acoustic Device-Class Classifier

Train a local classifier from labeled WAV recordings, evaluate it on a stratified holdout, and score later recordings. Bring your own recordings; this repository contains no generated audio or pre-trained model.

## Training data

Create one subfolder per label. Place WAV recordings for that label inside it, including nested folders if useful. The classifier reads PCM WAV files, converts stereo to mono, and extracts waveform and spectral features. Keep recording conditions consistent between training and prediction because microphone, room, distance, and sample rate can affect the signal.

```text
recordings/
  label_a/
    recording_001.wav
  label_b/
    recording_001.wav
```

The folder structure above describes the required schema; the repository does not include those files.

## Train and evaluate

Requires Python 3.9 or newer.

```bash
python -m pip install -r requirements.txt
python -m src.train --data-dir path/to/recordings --model acoustic_model.joblib --metrics acoustic_metrics.json
```

The trainer extracts RMS, peak, zero-crossing rate, spectral centroid, bandwidth, rolloff, entropy, dominant frequency, and dominant-power share. It fits a class-weighted logistic regression model, reports accuracy, balanced accuracy, macro F1, and a confusion matrix on a stratified holdout, then saves a model fitted on all supplied recordings.

Use `--test-size` and `--seed` to change the holdout. Every class needs at least two WAV files. Holdout results are specific to your data; a file-level random split can overstate performance when files from the same recording session appear on both sides. For a stronger evaluation, separate sessions or collection days before training.

## Predict

Score one WAV file or a folder of WAV files:

```bash
python -m src.predict --input path/to/recording.wav --model acoustic_model.joblib --output acoustic_predictions.csv
```

The output contains the predicted training label and a model score for each known class. The classifier always selects among the labels in its training data; it does not automatically recognize an unknown class, and its scores are not calibrated confidence guarantees.

## Scope and privacy

This is an offline recording classifier, not a live microphone monitor. Use recordings collected with the knowledge and permission of the people involved. It classifies the labels in your own dataset; it does not establish the presence of a specific device model or a person. The code makes no network requests. Only load model artifacts you trust.

## License

MIT. See [LICENSE](LICENSE).
