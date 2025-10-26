import streamlit as st
import numpy as np
import os
import random
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
import matplotlib.pyplot as plt
import io
import json
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score, confusion_matrix, precision_score, recall_score

# --- SETTINGS ---
MODEL_PATH = 'best_mobilenet_depression_model.h5'
IMG_SIZE = 224
TEST_DIR = 'depression_data/test'
CLASSES = ['Depressed', 'Not_Depressed']

# --- LOAD MODEL ---
def build_mobilenetv2_model():
    # recreate the same architecture used during training
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    # freeze lower layers, allow top 30 layers to train (same as training script)
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model


@st.cache_resource
def get_model():
    # Safer loading: avoid deserializing model config (which can fail across TF/Keras versions).
    # Rebuild the architecture and then load weights from the HDF5 file if possible.
    try:
        m = build_mobilenetv2_model()
    except Exception as e:
        raise RuntimeError(f"Failed to build MobileNetV2 architecture: {e}")

    # Try to load weights directly from the HDF5 file. This avoids deserializing InputLayer configs.
    try:
        import h5py
        with h5py.File(MODEL_PATH, 'r') as f:
            # common HDF5 keys when saving full model/weights
            has_model_weights = 'model_weights' in f
            has_layer_names = 'layer_names' in f

        if has_model_weights or has_layer_names:
            try:
                m.load_weights(MODEL_PATH)
                return m
            except Exception:
                # try by_name as a last-ditch (works if layer names match)
                m.load_weights(MODEL_PATH, by_name=True)
                return m
        else:
            # If HDF5 doesn't look like weight file, try the safe load_model as last resort
            try:
                return load_model(MODEL_PATH, compile=False)
            except Exception as e:
                raise RuntimeError(f"Model file did not contain weights and load_model also failed: {e}")

    except OSError as e:
        raise RuntimeError(f"Model file not found or unreadable at {MODEL_PATH}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed while attempting to load weights from {MODEL_PATH}: {e}")


model = get_model()

# --- PREDICTION FUNCTION ---
def predict_image(image, model):
    img = image.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    proba = model.predict(img_array, verbose=0)[0][0]
    prob_not_depressed = float(proba)
    prob_depressed = float(1 - proba)
    predicted_class = 'Not Depressed' if prob_not_depressed >= 0.5 else 'Depressed'
    confidence = max(prob_not_depressed, prob_depressed)
    margin = abs(prob_not_depressed - 0.5)
    uncertain = margin < 0.15
    return {
        'prediction': predicted_class,
        'confidence': confidence,
        'prob_depressed': prob_depressed,
        'prob_not_depressed': prob_not_depressed,
        'margin': margin,
        'uncertain': uncertain
    }

# --- PLOT RESULT ---
def render_prediction_widgets(st_container, image, result):
    """Render interactive prediction widgets inside given Streamlit container.
    Uses Plotly for nicer charts and layout.
    """
    # If Plotly is not available, fall back to a matplotlib image + plot_result display
    if not PLOTLY_AVAILABLE:
        # place a matplotlib summary image using the older plot_result Matplotlib function
        buf = plot_result_matplotlib(image, result)
        with st_container:
            st.image(buf, caption='Prediction Details (matplotlib)', use_column_width=True)
        return

    col_img, col_charts = st_container.columns([1, 2])

    # show image with rounded border
    with col_img:
        st.image(image, use_column_width=True, caption='Input image')
        if result['uncertain']:
            st.warning('Model is uncertain about this prediction — treat cautiously.')

    # build probability bar (Plotly)
    probs = {
        'Depressed': result['prob_depressed'],
        'Not Depressed': result['prob_not_depressed']
    }
    fig = go.Figure(go.Bar(
        x=list(probs.values()),
        y=list(probs.keys()),
        orientation='h',
        marker=dict(color=['#ff6b6b','#51cf66']),
        text=[f"{v:.3f}" for v in probs.values()],
        textposition='outside'
    ))
    fig.update_layout(title_text='Calibrated probabilities', margin=dict(l=20,r=20,t=30,b=10), xaxis=dict(range=[0,1]))

    # confidence gauge using indicator
    gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=result['confidence'],
        number={'valueformat': '.3f'},
        gauge={'axis': {'range': [0,1]}, 'bar': {'color': '#51cf66' if result['prediction']=='Not Depressed' else '#ff6b6b'}},
        title={'text': f"Prediction: {result['prediction']}"}
    ))
    gauge.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=250)

    # margin meter
    margin_val = result['margin']
    meter_fig = px.bar(x=['Certain','Borderline'], y=[1 - margin_val/0.5, margin_val/0.5], color=['Certain','Borderline'], color_discrete_sequence=['#4dabf7','#ffa94d'])
    meter_fig.update_layout(title_text=f'Margin {margin_val:.3f}', showlegend=False, yaxis=dict(range=[0,1]), margin=dict(l=10,r=10,t=30,b=10), height=250)

    with col_charts:
        st.plotly_chart(fig, use_container_width=True)
        cols = st.columns([1,1])
        with cols[0]:
            st.plotly_chart(gauge, use_container_width=True)
        with cols[1]:
            st.plotly_chart(meter_fig, use_container_width=True)


def plot_result_matplotlib(image, result):
    """Backward-compatible matplotlib renderer used when Plotly isn't available."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    # Image
    axes[0].imshow(image)
    axes[0].set_title('Input')
    axes[0].axis('off')
    # Probabilities
    axes[1].bar(['Depressed', 'Not Depressed'], [result['prob_depressed'], result['prob_not_depressed']], color=['#ff6b6b','#51cf66'], alpha=0.85)
    axes[1].set_ylim(0,1)
    axes[1].set_title('Calibrated Probabilities')
    for i, v in enumerate([result['prob_depressed'], result['prob_not_depressed']]):
        axes[1].text(i, v+0.02, f'{v:.3f}', ha='center', fontweight='bold')
    # Prediction
    bar_color = '#51cf66' if result['prediction']=='Not Depressed' else '#ff6b6b'
    axes[2].bar([result['prediction']], [result['confidence']], color=bar_color, alpha=0.85)
    axes[2].set_ylim(0,1)
    axes[2].set_title(result['prediction'] + (' (?)' if result['uncertain'] else ''))
    axes[2].text(0, result['confidence']+0.02, f"{result['confidence']:.3f}", ha='center', fontweight='bold')
    # Uncertainty
    levels = ['Certain','Borderline']
    meter_vals = [1 - result['margin']/0.5, result['margin']/0.5]
    meter_vals = [min(max(v,0),1) for v in meter_vals]
    axes[3].bar(levels, meter_vals, color=['#4dabf7','#ffa94d'], alpha=0.6)
    axes[3].set_ylim(0,1)
    axes[3].set_title(f"Margin {result['margin']:.3f}")
    if result['uncertain']:
        axes[3].text(1, meter_vals[1]+0.02, 'Uncertain', ha='center', fontweight='bold', color='#d9480f')
    for ax in axes:
        ax.set_xticks([])
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf


# --- STREAMLIT UI (improved) ---
st.set_page_config(page_title='SombriaAI — Illuminating Emotions Beyond the Surface', layout='wide', initial_sidebar_state='expanded')

# Header
st.markdown("""
<div style='display:flex; align-items:center; gap:12px'>
    <h1 style='margin:0'>SombriaAI — Illuminating Emotions Beyond the Surface</h1>
    <div style='color:#6c757d'> — Upload an image or run tests on the stored test-set</div>
</div>
""", unsafe_allow_html=True)

# Custom CSS for nicer cards and images
st.markdown("""
<style>
.thumb-card img {border-radius:12px; border:3px solid rgba(255,255,255,0.06); box-shadow:0 6px 16px rgba(0,0,0,0.12);} 
.demo-grid {display:flex; gap:12px; overflow-x:auto; padding:6px 0 18px 0}
.demo-item {flex:0 0 auto; width:140px}
.header-note {color:#6c757d; font-size:14px}
</style>
""", unsafe_allow_html=True)

# Small thumbnail gallery (click to load into Single test image tab)
demo_candidates = []
if os.path.exists(TEST_DIR):
    for cls in CLASSES:
        folder = os.path.join(TEST_DIR, cls)
        if os.path.exists(folder):
            files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
            demo_candidates += files
demo_candidates = random.sample(demo_candidates, min(len(demo_candidates), 8)) if demo_candidates else []
if demo_candidates:
    st.markdown('**Try a demo image** — click any thumbnail to load it into the selector below')
    cols = st.columns(len(demo_candidates))
    for i, p in enumerate(demo_candidates):
        with cols[i]:
            try:
                st.image(load_img(p), caption=os.path.basename(p), width=120)
                if st.button('Use', key=f'demo_{i}'):
                    st.session_state['demo_img_path'] = p
            except Exception:
                pass
    st.markdown('---')

# Sidebar controls
with st.sidebar:
    st.header('Controls')
    strategy = st.selectbox('Decision strategy', ['argmax', 'threshold'])
    threshold = st.slider('Threshold (only used if strategy=threshold)', 0.01, 0.99, 0.5, 0.01)
    show_plots = st.checkbox('Show detailed plots', value=True)
    fun_mode = st.checkbox('Enable fun mode (balloons for confident predictions)', value=False)
    st.markdown('---')
    st.subheader('Model info')
    if os.path.exists('mobilenet_model_metrics.json'):
        try:
            with open('mobilenet_model_metrics.json','r') as f:
                metrics = json.load(f)
            st.write(f"Type: {metrics.get('model_type')}")
            st.write(f"Image size: {metrics.get('image_size')}")
            st.write(f"Val accuracy: {metrics.get('accuracy')}")
        except Exception:
            st.write('Could not read mobilenet_model_metrics.json')
    else:
        st.write('Using saved model file')

    # --- About / Author ---
    st.markdown('---')
    st.subheader('About the author')
    st.markdown(
        """
        **Taksheel Saini**  
        AI/ML Privacy Product Enthusiast  

        - Email: [tak​sheel13@gmail.com](mailto:tak​sheel13@gmail.com)  
        - GitHub: [github.com/taksheelsaini](https://github.com/taksheelsaini)  
        - LinkedIn: [linkedin.com/in/taksheelsaini](https://www.linkedin.com/in/taksheelsaini)
        """,
        unsafe_allow_html=True,
    )

# Tabs
tab1, tab2, tab3 = st.tabs(["Single test image", "Upload image", "Batch test (test set)"])

with tab1:
    st.subheader('Pick an image from your test set')
    left, right = st.columns([1,2])
    with left:
        class_choice = st.selectbox('Choose class', CLASSES)
        test_folder = os.path.join(TEST_DIR, class_choice)
        images = []
        if os.path.exists(test_folder):
            images = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        if images:
            demo_path = st.session_state.get('demo_img_path') if 'demo_img_path' in st.session_state else None
            if demo_path and os.path.dirname(demo_path) == test_folder and os.path.basename(demo_path) in images:
                img_file = st.selectbox('Pick a test image', images, index=images.index(os.path.basename(demo_path)))
            else:
                img_file = st.selectbox('Pick a test image', images)
            img_path = os.path.join(test_folder, img_file)
        else:
            st.info('No images found in this class/folder')
            img_file = None
    with right:
        if img_file:
            image = load_img(img_path)
            st.image(image, caption=f"Selected: {img_file}", width=320)
            if st.button('Predict this test image'):
                with st.spinner('Running prediction...'):
                    result = predict_image(image, model)
                st.success(f"Prediction: {result['prediction']} (Confidence {result['confidence']:.3f})")
                if result['uncertain']:
                    st.warning('Model is uncertain about this prediction.')
                if fun_mode and result['confidence']>=0.90:
                    st.balloons()
                if show_plots:
                    render_prediction_widgets(st.container(), image, result)
                    with st.expander('Download prediction details'):
                        st.download_button('Download JSON', data=json.dumps(result, indent=2), file_name='prediction.json')

with tab2:
    st.subheader('Upload a new image')
    uploaded_file = st.file_uploader('Drag an image here or click to browse', type=['png','jpg','jpeg'])
    if uploaded_file is not None:
        image = load_img(uploaded_file)
        st.image(image, caption='Uploaded Image', width=320)
        if st.button('Predict uploaded image'):
            with st.spinner('Running prediction...'):
                result = predict_image(image, model)
            st.success(f"Prediction: {result['prediction']} (Confidence {result['confidence']:.3f})")
            if result['uncertain']:
                st.warning('Model is uncertain about this prediction.')
            if fun_mode and result['confidence']>=0.90:
                st.balloons()
            if show_plots:
                render_prediction_widgets(st.container(), image, result)
                with st.expander('Download prediction details'):
                    st.download_button('Download JSON', data=json.dumps(result, indent=2), file_name='prediction.json')

with tab3:
    st.subheader('Run batch evaluation on the test set')
    if st.button('Run full test-set evaluation'):
        test_images = []
        for cls in CLASSES:
            folder = os.path.join(TEST_DIR, cls)
            if os.path.exists(folder):
                found = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]
                test_images += [(p, 0 if cls=='Depressed' else 1) for p in found]

        if not test_images:
            st.info('No test images found to evaluate.')
        else:
            y_true = []
            y_pred = []
            probs = []
            progress = st.progress(0)
            n = len(test_images)
            for i, (p, label) in enumerate(test_images):
                img = load_img(p, target_size=(IMG_SIZE, IMG_SIZE))
                res = predict_image(img, model)
                proba = res['prob_not_depressed']
                pred = 1 if (strategy=='argmax' and proba>=0.5) or (strategy=='threshold' and proba>=threshold) else 0
                y_true.append(label)
                y_pred.append(pred)
                probs.append(proba)
                progress.progress(int((i+1)/n*100))

            # metrics
            acc = accuracy_score(y_true, y_pred)
            bal = balanced_accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred)

            st.metric('Accuracy', f"{acc:.3f}")
            st.metric('Balanced Acc', f"{bal:.3f}")
            st.metric('F1 score', f"{f1:.3f}")
            st.write('Confusion matrix:')
            st.write(cm.tolist())

            results = {'accuracy': acc, 'balanced_accuracy': bal, 'f1': f1, 'precision': prec, 'recall': rec, 'confusion_matrix': cm.tolist()}
            st.download_button('Download results (json)', data=json.dumps(results, indent=2), file_name='test_results.json')
