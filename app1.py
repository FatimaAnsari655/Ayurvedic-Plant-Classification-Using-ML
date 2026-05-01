import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from Plants_information import plant_info

# Page Config

st.set_page_config(
    page_title=" AI Leaf Classification System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Premium CSS Styling
# -----------------------------
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Poppins:wght@300;400;500;600&family=Inter:wght@400;500;600&display=swap');

/* Global Styles */
.stApp {
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 25%, #c8e6c9 50%, #a5d6a7 75%, #81c784 100%);
    background-attachment: fixed;
    font-family: 'Poppins', sans-serif;
    color: #1b5e20;
}

/* Hide default Streamlit elements */
.stDeployButton, .stHeader {
    display: none;
}

/* Custom Container Styles */
.glass-card {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    padding: 2rem;
    margin: 1rem 0;
    transition: all 0.3s ease;
}

.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 40px 0 rgba(31, 38, 135, 0.2);
}

/* Hero Section */
.hero-section {
    background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 50%, #0d4f0d 100%);
    border-radius: 25px;
    padding: 3rem 2rem;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
    position: relative;
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
}

.hero-section::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: float 15s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translate(0, 0) rotate(0deg); }
    50% { transform: translate(-30px, -30px) rotate(180deg); }
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.hero-subtitle {
    font-size: 1.3rem;
    opacity: 0.95;
    margin: 1rem 0 0;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* Section Headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    color: #1b5e20;
    margin: 1.5rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Upload Area */
.upload-area {
    border: 2px dashed #81c784;
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    background: rgba(255, 255, 255, 0.5);
    transition: all 0.3s ease;
}

.upload-area:hover {
    border-color: #4caf50;
    background: rgba(255, 255, 255, 0.7);
}

/* Custom Buttons */
.premium-btn {
    background: linear-gradient(135deg, #4caf50 0%, #2e7d32 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-family: 'Poppins', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
}

.premium-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
}

/* Result Cards */
.result-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 2rem;
    border-left: 5px solid #4caf50;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    margin: 1rem 0;
}

.confidence-badge {
    background: linear-gradient(135deg, #66bb6a 0%, #4caf50 100%);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-weight: 600;
    display: inline-block;
    margin: 0.5rem 0;
}

/* Plant Info Grid */
.info-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.info-item {
    background: rgba(255, 255, 255, 0.6);
    padding: 1rem;
    border-radius: 12px;
    border-left: 3px solid #81c784;
}

.info-label {
    font-weight: 600;
    color: #2e7d32;
    margin-bottom: 0.3rem;
}

.info-value {
    color: #1b5e20;
    font-size: 0.95rem;
}

/* Metrics Section */
.metrics-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-card {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.8) 0%, rgba(255, 255, 255, 0.6) 100%);
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(129, 199, 132, 0.3);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #2e7d32;
    margin-bottom: 0.5rem;
}

.metric-label {
    color: #4caf50;
    font-weight: 500;
}

/* Benefits List */
.benefits-list {
    list-style: none;
    padding: 0;
}

.benefits-list li {
    background: rgba(255, 255, 255, 0.5);
    margin: 0.5rem 0;
    padding: 0.8rem 1rem;
    border-radius: 10px;
    border-left: 3px solid #66bb6a;
    transition: all 0.3s ease;
}

.benefits-list li:hover {
    background: rgba(255, 255, 255, 0.8);
    transform: translateX(5px);
}

/* Image Display */
.plant-image {
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    margin: 1rem 0;
}

/* Search Input */
.search-input {
    background: rgba(255, 255, 255, 0.8);
    border: 2px solid rgba(129, 199, 132, 0.5);
    border-radius: 12px;
    padding: 1rem;
    font-family: 'Poppins', sans-serif;
    transition: all 0.3s ease;
}

.search-input:focus {
    border-color: #4caf50;
    background: rgba(255, 255, 255, 0.95);
    box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
}

/* Loading Animation */
.loading-spinner {
    text-align: center;
    padding: 2rem;
}

.leaf-icon {
    display: inline-block;
    animation: sway 3s ease-in-out infinite;
}

@keyframes sway {
    0%, 100% { transform: rotate(-5deg); }
    50% { transform: rotate(5deg); }
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .hero-title {
        font-size: 2.5rem;
    }
   
    .hero-subtitle {
        font-size: 1.1rem;
    }
   
    .info-grid {
        grid-template-columns: 1fr;
    }
   
    .metrics-container {
        grid-template-columns: 1fr;
    }
}

/* Custom Streamlit Elements Override */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.8);
    border: 2px solid rgba(129, 199, 132, 0.5);
    border-radius: 12px;
    padding: 0.8rem;
}

.stFileUploader {
    background: rgba(255, 255, 255, 0.5);
    border: 2px dashed #81c784;
    border-radius: 15px;
    padding: 1rem;
}

.stSuccess {
    background: linear-gradient(135deg, rgba(102, 187, 106, 0.1) 0%, rgba(76, 175, 80, 0.1) 100%);
    border-left: 4px solid #4caf50;
    border-radius: 10px;
    padding: 1rem;
}

.stWarning {
    background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
    border-left: 4px solid #ff9800;
    border-radius: 10px;
    padding: 1rem;
}

.stInfo {
    background: linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(3, 169, 244, 0.1) 100%);
    border-left: 4px solid #2196f3;
    border-radius: 10px;
    padding: 1rem;
}
/* Fix Search Plant title */
.section-header {
    color: #0d3b0d !important;
    font-weight: 700;
}

/* Fix input label text */
.stTextInput label {
    color: #0d3b0d !important;
    font-weight: 600 !important;
}

/* Fix input box text */
.stTextInput input {
    color: #000000 !important;
    background-color: #ffffff !important;
}

/* Fix placeholder text */
.stTextInput input::placeholder {
    color: #555555 !important;
    opacity: 1 !important;
}

/* Fix OR text */
div:has(> .stFileUploader) {
    color: #0d3b0d !important;
}

/* Fix error message visibility */
.stError {
    background: #ffe6e6 !important;
    color: #b30000 !important;
    font-weight: 600;
}

/* Fix spinner text */
.stSpinner > div {
    color: #0d3b0d !important;
    font-weight: 600;
}
/* Improve error message visibility */
.stError {
    background-color: #ffe5e5 !important;   /* light red background */
    border-left: 6px solid #d32f2f !important;
    color: #8b0000 !important;              /* dark red text */
    font-weight: 600 !important;
    border-radius: 10px !important;
}

/* Fix text inside error message */
.stError div {
    color: #8b0000 !important;
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Hero Header
# -----------------------------
st.markdown("""
<div class="hero-section">
    <div style="position: relative; z-index: 2;">
        <h1 class="hero-title">
            <span class="leaf-icon">🌿</span> AI Leaf Classification System
        </h1>
        <p class="hero-subtitle">
            Discover the Healing Power of Medicinal Plants with Advanced AI Technology
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_model_final.keras")

# Plant labels dictionary
labels = {
    0:'Aloevera', 1:'Amla', 2:'Anive-Dantu', 3:'Basale', 4:'Betel',
    5:'Bhrami', 6:'Coriender', 7:'Crape_Jasmine', 8:'Curry',
    9:'Drumstick', 10:'Fenugreek', 11:'Guava', 12:'Henna',
    13:'Hibiscus', 14:'Honge', 15:'Indian_Beech',
    16:'Indian_Mustard', 17:'Insulin', 18:'Jackfruit',
    19:'Jamaica_Cherry-Gasagase', 20:'Jamun', 21:'Jasmine',
    22:'Karanda', 23:'Lemon', 24:'Mango', 25:'Marigold',
    26:'Mexican_Mint', 27:'Mint', 28:'Neem', 29:'Oleander',
    30:'Palak(Spinach)', 31:'Papaya', 32:'Parijata',
    33:'Peepal', 34:'Pomegranate', 35:'Rasna',
    36:'Rose', 37:'Rose_apple', 38:'Roxburgh_fig',
    39:'Sandalwood', 40:'Seethapala', 41:'Tamarind',
    42:'Tulsi', 43:'Turmeric', 44:'Ashoka'
}

# -----------------------------
# Main Layout
# -----------------------------
left_col, right_col = st.columns([1, 1.2], gap="large")

# LEFT: Upload Panel
with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
   
    # Search Section
    st.markdown('<h2 class="section-header">🔍 Search Plant</h2>', unsafe_allow_html=True)
    search_name = st.text_input(
        "Enter plant name (e.g., Neem, Tulsi, Ashoka)",
        key="search_input",
        placeholder="Type and press Enter..."
    ).strip()
   
    st.markdown('<div style="margin: 2rem 0; text-align: center; color: #4caf50; font-weight: 500;">— OR —</div>', unsafe_allow_html=True)
   
    # Upload Section
    st.markdown('<h2 class="section-header">📤 Upload Image</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose a plant image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the plant for AI classification"
    )
   
    st.markdown('</div>', unsafe_allow_html=True)

# RIGHT: Results Panel
with right_col:
    plant_name = None
    info = None
    confidence = 0

    # SEARCH MODE
    if search_name:
        for plant in plant_info:
            if plant.lower() == search_name.lower():
                plant_name = plant
                info = plant_info[plant]
                break

        if plant_name:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown(f'<h2 style="color: #2e7d32; margin: 0 0 1rem 0;">🌱 {plant_name}</h2>', unsafe_allow_html=True)
           
            # Plant Information Grid
            st.markdown('<div class="info-grid">', unsafe_allow_html=True)
            st.markdown(f'''
            <div class="info-item">
                <div class="info-label">Scientific Name</div>
                <div class="info-value"><em>{info['scientific_name']}</em></div>
            </div>
            <div class="info-item">
                <div class="info-label">Family</div>
                <div class="info-value">{info['family']}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Parts Used</div>
                <div class="info-value">{', '.join(info['parts_used'])}</div>
            </div>
            ''', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
           
            # Ayurvedic Properties
            st.markdown('<h3 style="color: #2e7d32; margin: 1.5rem 0 1rem;">🌿 Ayurvedic Properties</h3>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                rasa = info['ayurvedic_properties'].get('rasa', 'Not specified')
                virya = info['ayurvedic_properties'].get('virya', 'Not specified')
                st.markdown(f'''
                <div class="info-item">
                    <div class="info-label">Rasa (Taste)</div>
                    <div class="info-value">{rasa}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Virya (Energy)</div>
                    <div class="info-value">{virya}</div>
                </div>
                ''', unsafe_allow_html=True)
            with col2:
                vipaka = info['ayurvedic_properties'].get('vipaka', 'Not specified')
                guna = info['ayurvedic_properties'].get('guna', 'Not specified')
                st.markdown(f'''
                <div class="info-item">
                    <div class="info-label">Vipaka (Post-digestive)</div>
                    <div class="info-value">{vipaka}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">Guna (Quality)</div>
                    <div class="info-value">{guna}</div>
                </div>
                ''', unsafe_allow_html=True)
           
            # Health Benefits
            st.markdown('<h3 style="color: #2e7d32; margin: 1.5rem 0 1rem;">💚 Health Benefits</h3>', unsafe_allow_html=True)
            st.markdown('<ul class="benefits-list">', unsafe_allow_html=True)
            for benefit in info['benefits']:
                st.markdown(f'<li>🌿 {benefit}</li>', unsafe_allow_html=True)
            st.markdown('</ul>', unsafe_allow_html=True)
           
            st.markdown('</div>', unsafe_allow_html=True)

    # IMAGE UPLOAD MODE
    elif uploaded_file is not None:
        st.markdown('<div class="loading-spinner">', unsafe_allow_html=True)
        with st.spinner('🔍 Analyzing plant image with AI...'):
            try:
                model = load_model()
                img = image.load_img(uploaded_file, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)

                prediction = model.predict(img_array)
                class_index = np.argmax(prediction)
                confidence = float(np.max(prediction))
                # Check if image is a leaf
                if confidence < 0.60:    
                    st.error("❌ The uploaded image does not appear to be a medicinal plant leaf. Please upload a clear leaf image.")
                    st.stop()

                plant_name = labels[class_index]

                st.markdown('</div>', unsafe_allow_html=True)
               
                # Display uploaded image
                st.image(uploaded_file, caption="📸 Uploaded Plant Image", use_column_width=True, output_format="auto")
               
                # Prediction Result
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
               
                if confidence > 0.7:
                    st.markdown(f'<h2 style="color: #2e7d32; margin: 0 0 1rem 0;">🌱 {plant_name}</h2>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-badge">✨ {confidence*100:.1f}% Confidence</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<h2 style="color: #ff9800; margin: 0 0 1rem 0;">⚠️ {plant_name}</h2>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-badge" style="background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);">⚠️ {confidence*100:.1f}% Confidence</div>', unsafe_allow_html=True)

                if plant_name in plant_info:
                    info = plant_info[plant_name]
                   
                    # Quick Info
                    st.markdown('<div class="info-grid">', unsafe_allow_html=True)
                    st.markdown(f'''
                    <div class="info-item">
                        <div class="info-label">Scientific Name</div>
                        <div class="info-value"><em>{info['scientific_name']}</em></div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Family</div>
                        <div class="info-value">{info['family']}</div>
                    </div>
                    ''', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                   
                    # Description
                    st.markdown(f'<p style="color: #1b5e20; line-height: 1.6; margin: 1rem 0;">{info["description"]}</p>', unsafe_allow_html=True)
                   
                    # Key Benefits
                    st.markdown('<h3 style="color: #2e7d32; margin: 1.5rem 0 1rem;">🌿 Key Benefits</h3>', unsafe_allow_html=True)
                    st.markdown('<ul class="benefits-list">', unsafe_allow_html=True)
                    for benefit in info['benefits'][:3]:
                        st.markdown(f'<li>💚 {benefit}</li>', unsafe_allow_html=True)
                    st.markdown('</ul>', unsafe_allow_html=True)
               
                st.markdown('</div>', unsafe_allow_html=True)
               
            except Exception as e:
                st.markdown('</div>', unsafe_allow_html=True)
                st.error(f"❌ An error occurred: {str(e)}")

    else:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('''
        <div style="text-align: center; padding: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2e7d32; margin-bottom: 1rem;">Welcome to AI Plant Classification</h3>
            <p style="color: #4caf50; margin-bottom: 1.5rem;">Search for a plant by name or upload an image to get started</p>
            <div style="background: rgba(255, 255, 255, 0.6); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                <p style="color: #1b5e20; font-weight: 500;">Try searching for:</p>
                <p style="color: #4caf50; font-style: italic;">Neem, Tulsi, Ashoka, Turmeric, Aloe Vera</p>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Accuracy Metrics Section (if prediction made)
# -----------------------------
if uploaded_file is not None and confidence > 0:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">📊 AI Model Performance</h2>', unsafe_allow_html=True)
   
    col1, col2, col3, col4 = st.columns(4)
   
    with col1:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value">{:.0f}%</div>
            <div class="metric-label">Confidence</div>
        </div>
        '''.format(confidence * 100), unsafe_allow_html=True)
   
    with col2:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value">45</div>
            <div class="metric-label">Plant Classes</div>
        </div>
        ''', unsafe_allow_html=True)
   
    with col3:
        st.markdown('''
        <div class="metric-card">
            <div class="metric-value">224×224</div>
            <div class="metric-label">Image Size</div>
        </div>
        ''', unsafe_allow_html=True)
   
    with col4:
        accuracy_level = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
        color = "#4caf50" if confidence > 0.8 else "#ff9800" if confidence > 0.6 else "#f44336"
        st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value" style="color: {color};">{accuracy_level}</div>
            <div class="metric-label">Accuracy Level</div>
        </div>
        ''', unsafe_allow_html=True)
   
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown('''
<div style="background: rgba(255, 255, 255, 0.8); border-radius: 15px; padding: 1.5rem; margin-top: 2rem; text-align: center; border: 1px solid rgba(129, 199, 132, 0.3);">
    <p style="color: #2e7d32; font-weight: 600; margin: 0;">🌿 AI Plant Classification System</p>
    <p style="color: #4caf50; font-size: 0.9rem; margin: 0.5rem 0 0;">Bridging Traditional Knowledge with Modern AI Technology</p>
    <p style="color: #81c784; font-size: 0.8rem; margin: 0.5rem 0 0;">© 2024 | For Educational Purposes Only</p>
</div>
''', unsafe_allow_html=True) 