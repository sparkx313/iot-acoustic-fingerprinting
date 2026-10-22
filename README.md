# IoT Acoustic Device Fingerprinting for Network Security

> Identify specific IoT devices (smart bulb vs camera vs speaker) purely from their electromagnetic hum and acoustic signatures — zero network access required.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Concept

Every electronic device produces a **unique acoustic fingerprint**:
- **Coil whine**: switching power supplies emit ultrasonic tones at 10-80 kHz
- **Fan signatures**: cooling fans have blade-count-specific frequency patterns
- **Capacitor hum**: aging capacitors produce distinctive low-frequency noise
- **Vibration patterns**: hard drives, speakers, motors have mechanical resonances

This system uses a microphone (or accelerometer) + deep learning to identify **which specific device model is running** in a room — without connecting to it, scanning its network, or accessing any API.

---

## Threat & Defense Context

**Privacy threat**: An attacker with a microphone could determine:
- Whether a smart speaker (Alexa/Google Home) is in the room
- If a security camera is active (motor drive signatures)
- What model of laptop/device the user has (coil whine patterns)

**Defensive use cases**:
- Corporate network security: detect rogue/unauthorized IoT devices physically present
- Smart home privacy: verify no unknown listening devices are active
- Forensics: identify devices present in a room during an incident

---

## Model Architecture

```
Raw Audio (48kHz) → STFT (2048-window, 512-hop)
                 → Mel Spectrogram (128 bins, 0-24kHz)
                 → SpecAugment training
                 → ResNet-18 (modified 1-channel input)
                 → Device Class (15 device categories)
                 → Confidence Score
```

**Device Classes:**
| Class | Typical Source |
|-------|---------------|
| SmartSpeaker_ Alexa | Switching PSU + mic array hum |
| SmartSpeaker_Google | Different PSU topology |
| SecurityCamera | IR-cut filter motor + fan |
| SmartBulb_PhilipsHue | PWM driver signature |
| SmartPlug | Relay + capacitor |
| Router_WiFi | Multiple switching regulators |
| SmartTV_Backlight | LED driver coil whine |
| Thermostat | Compressor cycling pattern |
| Doorbell_Video | PIR sensor + speaker driver |
| Hub_Zigbee | RF oscillator leakage |
|... | (15 total classes in dataset) |

---

## Usage

### Collect Training Data

```bash
python -m src.audio_collector \
    --device-label "SecurityCamera" \
    --duration 300 \
    --sample-rate 48000 \
    --output data/raw/
```

### Train Fingerprint Classifier

```bash
python -m src.train \
    --data-dir data/processed/ \
    --model resnet18 \
    --epochs 60 \
    --batch-size 32 \
    --output checkpoints/iot_acoustic.pt
```

### Real-Time Detection

```bash
python -m src.detect_live \
    --model checkpoints/iot_acoustic.pt \
    --device-index 0 \
    --threshold 0.85 \
    --log-file detections.log
```

---

## Project Structure

```
iot-acoustic-fingerprinting/
├── src/
│   ├── audio_collector.py   # Microphone recording + labeling
│   ├── features.py          # STFT, mel spectrogram, MFCC
│   ├── model.py             # ResNet-18 classifier
│   ├── train.py             # Training loop
│   ├── detect_live.py       # Real-time inference stream
│   └── dataset.py           # PyTorch Dataset
├── data/
├── checkpoints/
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Accuracy (Expected)

| Scenario | Top-1 | Top-3 |
|----------|-------|-------|
| Single device, quiet room | **94%** | **99%** |
| Multiple devices overlapping | 78% | 92% |
| 30dB background noise | 67% | 85% |
| Different recording device | 61% | 79% |

---

## References

- **Acoustic side-channel attacks** (Acs & Castiglione, 2017)
- **Device fingerprinting via EMI** (Wu et al., USENIX Security, 2021)
- **Coil whine analysis** for power supply identification
- **ResNet** (He et al., 2015)

---

## License

MIT © [sparkx313](https://github.com/sparkx313)
