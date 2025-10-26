# SombriaAI — Illuminating Emotions Beyond the Surface ✨🖼️

Hi — I'm Taksheel Saini. I built SombriaAI to explore expressive signals in facial images and present them in a clear, interactive way. This repo contains the model, the utilities, and a Streamlit demo so you (or faculty) can demo the idea quickly and safely.

> SombriaAI is a research/demo project — it surfaces emotional signals from facial imagery for educational and product-research use. It is not medical software and must never be used for clinical diagnosis. See the full disclaimer below.

---

Table of contents
- About this project
- What I built (quick tour)
- Try it locally (quick start)
- Files you care about
- How the model works (short technical notes)
- Evaluation & metrics (what I measured)
- Design choices & privacy notes
- Development & reproducibility
- Contact / Author

---

About this project
------------------

I wanted to build a compact, demo-friendly system that can analyze facial imagery and present probabilistic outputs in a way that's intuitive, explainable, and fun to demo. SombriaAI focuses on emotion-derived signals (not clinical diagnosis). My goals were:

- Create a small, well-documented codebase that others can run locally.
- Provide a visually appealing UI that shows probabilities, uncertainty, and calibrated outputs.
- Keep the original research artifacts (notebook, training artifacts) but separate the demo/deployment code so nothing breaks during experimentation.

What I built (quick tour) 🚀

- A trained MobileNetV2-based classifier and helper scripts.
- A Streamlit demo (`app.py`) that:
    - Lets you pick test images from the repo or upload your own.
    - Shows calibrated probabilities, a confidence gauge, and an uncertainty meter.
    - Has a thumbnail gallery, downloadable JSON for predictions, and a batch-evaluation mode.
- Notebook(s) with the full training workflow and analysis (`depression_detection_project.ipynb`). I kept the notebook intact for reproducibility.

Quick start — run the Streamlit demo locally (2 minutes) ⏱️

1) (Optional) Create a virtual environment and activate it

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install the UI dependencies (recommended)

```bash
pip install -r requirements-ui.txt
```

3) Start the demo

```bash
streamlit run app.py
```

The demo will open in your browser. Use the left sidebar to choose threshold/strategy, enable `fun mode` for a playful effect, or run batch evaluation on the included test set.

Files you care about 📁

- `app.py` — Streamlit demo and the polished presentation layer (UI + model loader + Plotly visuals)
- `best_mobilenet_depression_model.h5` — saved model weights (kept as-is for reproducibility)
- `depression_detection_project.ipynb` — training & analysis notebook (kept unchanged)
- `depression_data/` — reorganized binary-style dataset used for validation/test (left intact)
- `mobilenet_model_metrics.json` — small JSON with model summary metrics
- `requirements-ui.txt` — dependencies for running the demo UI quickly

Note: I intentionally preserved dataset folder names and class labels so the training and demo code remain compatible with the original experiments.

How the model works — short technical notes 🔬

- Base backbone: MobileNetV2 pre-trained on ImageNet (transfer learning).
- Head: GlobalAveragePooling + Dense blocks with BatchNorm and Dropout, final sigmoid output for a binary decision probability.
- Input size: 224×224 RGB for the MobileNetV2-based demo model.
- Prediction semantics in the demo: the model outputs a probability (p) that is interpreted and displayed as two calibrated probabilities (class A / class B). The UI allows you to pick `argmax` or a `threshold` decision strategy and visualizes uncertainty using a simple margin heuristic.

Evaluation & metrics

I measured standard classification metrics on the test set and recorded them in `mobilenet_model_metrics.json`. The demo UI also provides a batch-run mode that recomputes accuracy, balanced accuracy, F1, precision, recall, and confusion matrix for any chosen decision strategy or threshold.

Design choices & privacy notes 🔐

- I intentionally kept all processing local. The Streamlit demo loads models and runs inference locally — there are no external API calls.
- SombriaAI is designed for research and product exploration. It is NOT medical software. Facial expressions are noisy and context-dependent — any inferences must be treated with caution.
- If you demo this project publicly, ensure you have consent to use any images, and prefer anonymized or synthetic images where possible.

Reproducibility & development

- If you want a reproducible containerized demo, I can add a Dockerfile. Let me know and I will add one that pins TensorFlow + Streamlit versions.
- The training notebook and the saved weights are both included. If you want the model exported to TensorFlow SavedModel format or re-saved under a particular TF/Keras version, I can provide a tiny helper script to run in the original training environment.

Author / Contact

By — Taksheel Saini  
AI/ML Privacy Product Enthusiast

- Email: taksheel13@gmail.com
- GitHub: https://github.com/taksheelsaini
- LinkedIn: https://www.linkedin.com/in/taksheelsaini

If you'd like the author bio on the UI or the README to show a picture or link to a public demo, tell me which image or URL to use and I will wire it up.

Disclaimer (important) ⚠️

This project is for educational and research/demo purposes only. It is not a diagnostic tool and must never be used for clinical decision-making. The models are trained on limited data and are subject to bias, distribution shift, and other failure modes.

---

Want it prettier? I can:
- Add a GIF header or a Lottie animation into the README for GitHub rendering (you’ll get an eye-catching animation).
- Add a Dockerfile and a single-command `run-demo.sh` so visitors can start the demo reliably in a container.
- Replace dataset folder names with SombriaAI-branded names everywhere (I avoided renaming on purpose because it changes paths used in code).

Tell me which of the above (GIF header / Docker / rename dataset folders / save model in SavedModel format) you want next and I’ll apply it. 

Happy to keep polishing — we can make this demo production-ready or keep it a research showcase depending on your goals. 😄
tensorflow>=2.20.0
