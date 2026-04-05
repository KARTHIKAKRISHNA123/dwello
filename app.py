import streamlit as st
import pandas as pd
import pickle
import time
import datetime

# 1. Page Configuration
st.set_page_config(page_title="HomeVista Predictor", layout="wide", initial_sidebar_state="collapsed")

# 2. Vercel Custom CSS & Animations
vercel_style = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Apply Vercel Font and Base Styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Fade-in animation for the main container */
    @keyframes fadeIn {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    .block-container {
        animation: fadeIn 0.8s ease-out;
        padding-top: 3rem;
        max-width: 900px;
    }

    /* Vercel Style Headers */
    h1 {
        font-weight: 700 !important;
        letter-spacing: -0.05em !important;
        font-size: 3rem !important;
    }
    h3 {
        font-weight: 500 !important;
        letter-spacing: -0.02em !important;
        color: #888888 !important;
    }

    /* Smooth Input Fields */
    .stSelectbox div[data-baseweb="select"] > div, 
    .stNumberInput div[data-baseweb="input"] > div {
        border-radius: 8px !important;
        border: 1px solid #333 !important;
        background-color: #0A0A0A !important;
        transition: all 0.2s ease;
    }
    
    /* Input Field Focus Animation */
    .stSelectbox div[data-baseweb="select"] > div:focus-within, 
    .stNumberInput div[data-baseweb="input"] > div:focus-within {
        border-color: #fff !important;
        box-shadow: 0 0 0 1px #fff !important;
    }

    /* Vercel Style Predict Button */
    .stButton > button {
        background-color: #ededed !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.75rem 1.5rem !important;
        width: 100%;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 14px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Button Hover Animation */
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        background-color: #ffffff !important;
        box-shadow: 0 6px 20px 0 rgba(255, 255, 255, 0.2);
    }

    /* Subtle divider */
    hr {
        border-color: #333333 !important;
        margin: 3rem 0;
    }
</style>
"""
st.markdown(vercel_style, unsafe_allow_html=True)

# 3. Load the saved pipeline
@st.cache_resource 
def load_model():
    with open('house_price_model.pkl', 'rb') as file:  # Update to 'house_price_model.pkl' if that is your file name
        return pickle.load(file)

model = load_model()

# 4. Full Form Dictionaries
mssubclass_map = {
    '20': '1-Story (1946 & Newer)', '30': '1-Story (Older)', '40': '1-Story (Finished Attic)', 
    '45': '1.5-Story (Unfinished)', '50': '1.5-Story (Finished)', '60': '2-Story', 
    '70': '2-Story (Older)', '75': '2.5-Story', '80': 'Split or Multi-Level', 
    '85': 'Split Foyer', '90': 'Duplex', '120': '1-Story Planned Unit Dev', 
    '160': '2-Story Planned Unit Dev', '180': 'PUD Multilevel', '190': '2-Family Conversion'
}
mszoning_map = {'RL': 'Residential Low Density', 'RM': 'Residential Medium Density', 'C (all)': 'Commercial', 'FV': 'Floating Village', 'RH': 'Residential High Density'}
bldgtype_map = {'1Fam': 'Single-Family Detached', '2fmCon': 'Two-Family Conversion', 'Duplex': 'Duplex', 'TwnhsE': 'Townhouse End Unit', 'Twnhs': 'Townhouse Inside Unit'}
lotconfig_map = {'Inside': 'Inside Lot', 'Corner': 'Corner Lot', 'CulDSac': 'Cul-de-sac', 'FR2': 'Frontage 2 Sides', 'FR3': 'Frontage 3 Sides'}
exterior1st_map = {'VinylSd': 'Vinyl Siding', 'MetalSd': 'Metal Siding', 'Wd Sdng': 'Wood Siding', 'HdBoard': 'Hard Board', 'BrkFace': 'Brick Face', 'WdShing': 'Wood Shingles', 'CemntBd': 'Cement Board', 'Plywood': 'Plywood', 'AsbShng': 'Asbestos Shingles', 'Stucco': 'Stucco', 'BrkComm': 'Brick Common', 'AsphShn': 'Asphalt Shingles', 'Stone': 'Stone', 'ImStucc': 'Imitation Stucco', 'CBlock': 'Cinder Block'}

# 5. Build the UI
st.markdown("<h1>HomeVista Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter property specifications to generate an AI-driven valuation.</h3>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    mssubclass = st.selectbox("Building Class", options=list(mssubclass_map.keys()), format_func=lambda x: mssubclass_map[x])
    mszoning = st.selectbox("Zoning Classification", options=list(mszoning_map.keys()), format_func=lambda x: mszoning_map[x])
    bldgtype = st.selectbox("Dwelling Type", options=list(bldgtype_map.keys()), format_func=lambda x: bldgtype_map[x])
    lotconfig = st.selectbox("Lot Configuration", options=list(lotconfig_map.keys()), format_func=lambda x: lotconfig_map[x])
    exterior1st = st.selectbox("Exterior Material", options=list(exterior1st_map.keys()), format_func=lambda x: exterior1st_map[x])

with col2:
    lotarea = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=200000, value=10000)
    totalbsmtsf = st.number_input("Total Basement (sq ft)", min_value=0, max_value=6000, value=1000)
    bsmtfinsf2 = st.number_input("Type 2 Finished Bsmt (sq ft)", min_value=0, max_value=1500, value=0)
    overallcond = st.slider("Overall Condition", min_value=1, max_value=10, value=5)
    
    # We ask for the year, which is easier for the user to answer
    yearbuilt = st.number_input("Construction Year", min_value=1800, max_value=2024, value=2000)

st.markdown("---")

# 6. Predict Button with Animations
if st.button("Generate Valuation"):
    # Trigger a toast notification (slides in from bottom right)
    st.toast("Connecting to prediction engine...", icon="⚙️")
    
    with st.spinner("Analyzing property parameters..."):
        time.sleep(1.2) # Artificial delay to let the animation play out beautifully
        
        # -------------------------------------------------------------
        # FIX IMPLEMENTED HERE: Option 1
        # Calculate HouseAge dynamically before sending to the model!
        # -------------------------------------------------------------
        current_year = datetime.date.today().year
        calculated_house_age = current_year - yearbuilt
        
        input_data = {
            'MSSubClass': mssubclass,
            'MSZoning': mszoning,
            'LotArea': lotarea,
            'LotConfig': lotconfig,
            'BldgType': bldgtype,
            'OverallCond': overallcond,
            'HouseAge': calculated_house_age, # Send the computed feature!
            'Exterior1st': exterior1st,
            'BsmtFinSF2': bsmtfinsf2,
            'TotalBsmtSF': totalbsmtsf
        }
        
        # Convert dictionary to a Pandas DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Run the DataFrame through the pipeline to get the prediction
        predicted_price = model.predict(input_df)[0] 
        
    # Success State
    st.success(f"### Estimated Market Value: **${predicted_price:,.2f}**")
    st.balloons()