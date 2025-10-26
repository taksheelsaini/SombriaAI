# SombriaAI — Illuminating Emotions Beyond the Surface (Streamlit UI)

This repository includes `app.py` — a Streamlit interface that loads `best_mobilenet_depression_model.h5` and provides: 

- Single-image prediction (pick from test set or upload a new image)
- Interactive, attractive prediction visualization (Plotly) with a Matplotlib fallback
- Thumbnail demo gallery for quick trials
- Batch evaluation over the test set and a downloadable JSON report

How to run

1. (Optional) Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-ui.txt
```

2. Start the Streamlit app:

```bash
streamlit run app.py
```

Notes

- `plotly` is optional. If it's not installed, the app will fall back to a Matplotlib-based static image for prediction details.
- If you encounter model-loading errors related to Keras/TensorFlow versions, see the `get_model()` helper in `app.py` which attempts to load weights safely. If the HDF5 was saved in a different Keras/TensorFlow version, re-saving the model in the training environment to a SavedModel or re-exported HDF5 may be needed.

If you'd like, I can:

- Add a small Dockerfile to run the app reproducibly
- Wire a CI job to run a quick smoke test
- Re-export the `best_mobilenet_depression_model.h5` to a TF SavedModel if you can run a short helper script in the original training environment

---

## Author

By - **Taksheel Saini** - AI/ML Privacy Product Enthusiast

- Email ID: taksheel13@gmail.com
- Github: https://github.com/taksheelsaini
- Linkedin: https://www.linkedin.com/in/taksheelsaini
