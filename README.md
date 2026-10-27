# 🦷 Oral Disease Classification

An image classification app that predicts oral diseases from a photo of the mouth/teeth.

## 🔗 Live App
**https://oral-disease-classifier.streamlit.app**

## Classes
- Calculus
- Caries
- Gingivitis
- Mouth Ulcer
- Tooth Discoloration
- Hypodontia

## Model
- Transfer learning with **EfficientNetB0** (ImageNet weights), input size 224×224
- Compared against a custom CNN and DenseNet121
- Test accuracy: ~87%

## Run locally
```
pip install -r requirements.txt
python -m streamlit run app.py
```

## Files
| File | Purpose |
|------|---------|
| `app.py` | Streamlit app |
| `best_model.keras` | Trained model |
| `class_names.json` | Class labels |
| `requirements.txt` | Dependencies |

> For educational purposes only - not a medical diagnosis.
