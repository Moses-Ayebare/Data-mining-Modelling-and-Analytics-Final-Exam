import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Box Office Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
    <style>
        .main-header {
            font-size: 2.5em;
            color: #DC143C;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .metric-box {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .success-box {
            background-color: #d4edda;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #28a745;
        }
        .failure-box {
            background-color: #f8d7da;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #dc3545;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD MODEL & SCALER
# ============================================================

@st.cache_resource
def load_model():
    try:
        with open('Ayebare_best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.error("Error loading model. Make sure Ayebare_best_model.pkl is in the same directory.")
        return None

@st.cache_resource
def load_scaler():
    try:
        with open('Ayebare_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except:
        st.error("Error loading scaler. Make sure Ayebare_scaler.pkl is in the same directory.")
        return None

model = load_model()
scaler = load_scaler()

if model is None or scaler is None:
    st.stop()

# ============================================================
# TITLE & DESCRIPTION
# ============================================================

st.markdown("<div class='main-header'> Movie Box Office Success Predictor</div>", 
            unsafe_allow_html=True)

st.markdown("""
    <p style='text-align: center; color: #666; font-size: 1.1em;'>
        Predict whether a movie will achieve box office success using machine learning.
        <br>Powered by Random Forest model trained on 3,288 movies (1916-2016)
    </p>
""", unsafe_allow_html=True)

st.divider()

# ============================================================
# SIDEBAR: INPUT FEATURES
# ============================================================

with st.sidebar:
    st.header(" Movie Attributes")
    st.write("Enter your movie details below:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.slider(
            " Budget ($M)",
            min_value=1,
            max_value=300,
            value=50,
            step=5,
            help="Production budget in millions USD"
        )
        
        duration = st.slider(
            "⏱️ Runtime (minutes)",
            min_value=60,
            max_value=240,
            value=120,
            step=5,
            help="Film duration in minutes"
        )
        
        imdb_score = st.slider(
            " IMDB Rating",
            min_value=1.0,
            max_value=10.0,
            value=7.0,
            step=0.1,
            help="Expected critical reception"
        )
    
    with col2:
        cast_popularity = st.slider(
            " Cast Popularity",
            min_value=0,
            max_value=25,
            value=5,
            step=1,
            help="Total actor Facebook likes (millions)"
        )
        cast_pop_millions = cast_popularity * 1_000_000
        
        director_popularity = st.slider(
            " Director Popularity",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Director Facebook likes (millions)"
        )
        director_pop_millions = director_popularity * 1_000_000
        
        num_votes = st.slider(
            " Expected Engagement",
            min_value=10000,
            max_value=500000,
            value=100000,
            step=10000,
            help="Expected IMDB votes"
        )

# ============================================================
# MAIN CONTENT AREA
# ============================================================

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader(" Movie Summary")
    
    summary_data = {
        "Budget": f"${budget}M",
        "Duration": f"{duration} minutes",
        "IMDB Rating": f"{imdb_score}/10",
        "Cast Popularity": f"{cast_popularity}M likes",
        "Director Popularity": f"{director_popularity}M likes",
        "Expected Votes": f"{num_votes:,}"
    }
    
    for key, value in summary_data.items():
        st.write(f"**{key}:** {value}")

with col2:
    st.subheader(" Prediction Result")
    
    # Create feature array with EXACT 40 features in correct order
    try:
        # Create a pandas Series with all 40 features
        features_dict = {
            # Numerical features (16)
            'budget': float(budget),
            'duration': float(duration),
            'imdb_score': float(imdb_score),
            'num_voted_users': float(num_votes),
            'cast_total_facebook_likes': float(cast_pop_millions),
            'director_facebook_likes': float(director_pop_millions),
            'log_budget': np.log1p(budget),
            'log_votes': np.log1p(num_votes),
            'budget_per_minute': budget / (duration + 1),
            'budget_per_vote': budget / (num_votes + 1),
            'log_cast_popularity': np.log1p(cast_pop_millions),
            'log_director_popularity': np.log1p(director_pop_millions),
            'cast_director_ratio': (director_pop_millions + 1) / (cast_pop_millions + 1),
            'quality_engagement': imdb_score * np.log1p(num_votes),
            'review_sentiment_score': 0.5,
            'years_since_release': 0.0,
            
            # Categorical features - Content ratings (5)
            'rating_G': 0,
            'rating_PG': 0,
            'rating_PG-13': 0,
            'rating_R': 0,
            'rating_Unknown': 0,
            'rating_Unrated': 0,
            
            # Genres (8)
            'genre_drama': 0,
            'genre_comedy': 0,
            'genre_thriller': 0,
            'genre_romance': 0,
            'genre_action': 0,
            'genre_crime': 0,
            'genre_adventure': 0,
            'genre_horror': 0,
            
            # Decades (11)
            'decade_1920': 0,
            'decade_1930': 0,
            'decade_1940': 0,
            'decade_1950': 0,
            'decade_1960': 0,
            'decade_1970': 0,
            'decade_1980': 0,
            'decade_1990': 0,
            'decade_2000': 0,
            'decade_2010': 0,
        }
        
        # Feature order (MUST match training data)
        feature_order = [
            # Numerical (16)
            'budget', 'duration', 'imdb_score', 'num_voted_users',
            'cast_total_facebook_likes', 'director_facebook_likes',
            'log_budget', 'log_votes', 'budget_per_minute', 'budget_per_vote',
            'log_cast_popularity', 'log_director_popularity', 'cast_director_ratio',
            'quality_engagement', 'review_sentiment_score', 'years_since_release',
            
            # Ratings (6) - add one more
            'rating_G', 'rating_PG', 'rating_PG-13', 'rating_R', 'rating_Unknown', 'rating_Unrated',
            
            # Genres (8)
            'genre_drama', 'genre_comedy', 'genre_thriller', 'genre_romance',
            'genre_action', 'genre_crime', 'genre_adventure', 'genre_horror',
            
            # Decades (10)
            'decade_1920', 'decade_1930', 'decade_1940', 'decade_1950',
            'decade_1960', 'decade_1970', 'decade_1980', 'decade_1990',
            'decade_2000', 'decade_2010'
        ]
        
        # Verify we have exactly 40 features
        assert len(feature_order) == 40, f"Feature count mismatch: {len(feature_order)} != 40"
        
        # Create feature array in correct order
        features_array = np.array([[features_dict[f] for f in feature_order]])
        
        # Scale features
        features_scaled = scaler.transform(features_array)
        
        # Get prediction
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        success_prob = probability[1]
        
        # Display result
        if prediction == 1:
            st.markdown("""
                <div class='success-box'>
                    <h2 style='color: #28a745;'> LIKELY TO SUCCEED</h2>
                    <p style='font-size: 1.2em; color: #155724;'>This movie has strong box office potential!</p>
                </div>
            """, unsafe_allow_html=True)
            st.metric("Success Probability", f"{success_prob:.1%}", delta="Positive Outlook")
        else:
            st.markdown("""
                <div class='failure-box'>
                    <h2 style='color: #dc3545;'> RISKY PROSPECT</h2>
                    <p style='font-size: 1.2em; color: #721c24;'>This movie faces box office challenges.</p>
                </div>
            """, unsafe_allow_html=True)
            st.metric("Failure Risk", f"{1-success_prob:.1%}", delta="Negative Outlook")
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

# ============================================================
# MODEL INSIGHTS
# ============================================================

st.divider()

st.subheader(" What Drives Box Office Success?")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(" Top Factor", "Audience Engagement", help="Number of IMDB votes (13.1%)")

with col2:
    st.metric("Critical Factor", "Budget Scale", help="Production budget (7.6%)")

with col3:
    st.metric(" Quality Metric", "IMDB Score", help="Critical reception (3.7%)")

st.write("""
    The model learned that **box office success depends on:**
    
    1. **Audience Engagement** - Films that resonate generate more votes
    2. **Budget Efficiency** - Smart spending per minute of runtime
    3. **Critical Reception** - IMDB scores reflect quality perception
    4. **Star Power** - Actor/director popularity influences audience reach
    5. **Temporal Trends** - Release timing affects market conditions
""")

# ============================================================
# MODEL LIMITATIONS
# ============================================================

st.divider()

st.subheader("Model Limitations & Disclaimers")

st.info("""
    **Important:** This model is a TOOL, not a guarantee.
    
    The model achieved **80.55% accuracy** on historical data but cannot predict:
    - Pandemics or external crises
    - Viral marketing phenomena
    - Award season impact
    - Competitor releases
    - Last-minute changes in marketing strategy
    
    **Use this as ONE INPUT to decision-making, not the sole factor.**
""")

# ============================================================
# ABOUT
# ============================================================

st.divider()

st.subheader("About This Application")

st.write("""
    **Model Details:**
    - Algorithm: Random Forest Classifier
    - Training data: 3,288 movies (1916-2016)
    - Accuracy: 80.55%
    - Features: 40 engineered features
    - ROC-AUC: 89.42%
    
    **Built by:** Ayebare Moses
    **Course:** Data Mining, Modelling & Analytics (DSC8307)
""")