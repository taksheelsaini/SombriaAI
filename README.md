# SombriaAI — Illuminating Emotions Beyond the Surface ✨🖼️

Hello — I'm Taksheel Saini. I built SombriaAI to experiment with expressive signals in facial images and to present the outputs in a clear, interactive demo. I wrote this README and the demo UI; the content below reflects my goals, choices, and instructions for running and exploring the project locally.

> Important: SombriaAI is a research/demo project. It surfaces emotional signals from images for educational and product research. It is not a medical device and must not be used for clinical diagnosis.

---

Table of contents

- About this project
- What I built (quick tour)
- Quick start (how I run it locally)
- Files you care about
- How the model works (short technical notes)
- Evaluation & metrics
- Design choices & privacy notes
- Reproducibility & next steps
- Contact — me (Taksheel)

---

About this project
------------------

I created SombriaAI to explore how facial expressions can be translated into probabilistic emotion signals. My priorities were:

- Make a small, runnable codebase that others can run on a laptop.
- Build an interactive UI that communicates uncertainty and calibration clearly.
- Keep the research notebook and weights so experiments remain reproducible while keeping the demo code separate and stable.

What I built (quick tour) 🚀

- A MobileNetV2-based classifier and helper scripts.
- A Streamlit demo (`app.py`) that I use to:
  - Pick test images from the repo or upload my own.
  - View calibrated probabilities, a confidence gauge, and an uncertainty margin.
  - Run batch evaluation across the test images and download results as JSON.
- The training and analysis notebook (`depression_detection_project.ipynb`) with the full experimental workflow.

Quick start — how I run the demo locally (≈ 2 minutes)

1) (Optional) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2) Install the UI packages (recommended)

```bash
pip install -r requirements-ui.txt
```

3) Start the Streamlit demo

```bash
streamlit run app.py
```

Open the URL Streamlit shows in your browser. Use the sidebar to change strategy, threshold, or enable the playful `fun mode` when demoing.

Files you care about 📁

- `app.py` — Streamlit demo and UI code (model loading, plotting, and interaction)
- `best_mobilenet_depression_model.h5` — saved model weights (kept for reproducibility)
- `depression_detection_project.ipynb` — full training and analysis notebook
- `depression_data/` — dataset folders used for validation/test (kept as-is)
- `mobilenet_model_metrics.json` — model metrics summary
- `requirements-ui.txt` — quick install list for the demo UI

I kept dataset folder names and labels unchanged so scripts and notebooks stay compatible.

How the model works — short technical notes 🔬

- Backbone: MobileNetV2 pretrained on ImageNet.
- Head: GlobalAveragePooling → Dense blocks with BatchNorm + Dropout → final sigmoid for a binary probability.
- Input: 224×224 RGB images for the MobileNetV2-based demo.
- The demo shows probabilities for both classes, a confidence gauge, and a margin-based uncertainty indicator. Decision behavior can be `argmax` or a user-set threshold.

Evaluation & metrics

I computed standard metrics on the included test set and saved results to `mobilenet_model_metrics.json`. The demo also offers a batch mode that recomputes accuracy, balanced accuracy, F1, precision, recall, and the confusion matrix for any chosen threshold or strategy.

Design choices & privacy notes 🔐

- All processing happens locally in the demo — there are no external API calls.
- This project is for research and product exploration only. Observations from facial imagery are noisy and context-dependent; treat them as signals, not diagnoses.
- When demoing publicly, obtain consent for images and prefer anonymized or synthetic images where feasible.

Reproducibility & next steps

- If you want, I can add a Dockerfile to pin versions and make the demo fully reproducible. Tell me and I'll add it.
- I can also provide a helper script to re-export the model to TensorFlow SavedModel format if you need a different serialization format.

Contact — me

By — Taksheel Saini  
AI/ML Privacy Product Enthusiast

- Email: taksheel13@gmail.com
- GitHub: https://github.com/taksheelsaini
- LinkedIn: https://www.linkedin.com/in/taksheelsaini

Disclaimer (important) ⚠️

This project and its models are for educational and research purposes only. They are not diagnostic tools and must not be used for medical decision-making.
