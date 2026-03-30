import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ---------------- LOAD ----------------
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
df = pd.read_csv('zomato.csv')

st.set_page_config(page_title="DineSmart", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>

/* App Background */
.stApp {
    background: #f9f9f9;
    color: #0a192f;
    font-family: 'Segoe UI', sans-serif;
}

/* Title */
.title {
    text-align: center;
    font-size: 55px;
    font-weight: 700;
    color: #0072ff;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #555555;
    margin-bottom: 30px;
}

/* Feature Box */
.feature-box {
    background: #ffffff;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
    color: #0072ff;
    border: 1px solid #e0e0e0;
    transition: 0.3s;
}

.feature-box:hover {
    transform: translateY(-5px);
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

/* Cards */
.card {
    background: #ffffff;
    padding: 20px;
    border-radius: 15px;
    margin: 10px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    transition: 0.3s;
    color: #0a192f;
}

.card:hover {
    transform: scale(1.03);
}

/* Button */
button[kind="primary"] {
    background: #0072ff;
    color: white;
    border-radius: 10px;
    font-weight: bold;
    height: 3em;
}

/* Section Header */
.section {
    font-size: 24px;
    margin-top: 30px;
    margin-bottom: 15px;
    color: #333333;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<div class="title">🍽️ DineSmart</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Smart restaurant discovery powered by sentiment intelligence</div>', unsafe_allow_html=True)

# ---------------- FEATURES ----------------
st.markdown('<div class="section">✨ Key Features</div>', unsafe_allow_html=True)

f1, f2, f3, f4 = st.columns(4)

with f1:
    st.markdown('<div class="feature-box">🤖 AI Sentiment Analysis</div>', unsafe_allow_html=True)

with f2:
    st.markdown('<div class="feature-box">📍 Location-Based Filtering</div>', unsafe_allow_html=True)

with f3:
    st.markdown('<div class="feature-box">🍜 Cuisine Selection</div>', unsafe_allow_html=True)

with f4:
    st.markdown('<div class="feature-box">📊 Visual Insights</div>', unsafe_allow_html=True)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔍 Recommendations", "📊 Insights"])

# ---------------- SENTIMENT FUNCTION ----------------
def predict_sentiment(text):
    vec = vectorizer.transform([str(text)])
    return model.predict(vec)[0]

# ================= TAB 1 =================
with tab1:

    st.markdown('<div class="section">🔍 Find Your Perfect Restaurant</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        location = st.selectbox("📍 Select Location", sorted(df['location'].dropna().unique()))

    with col2:
        cuisine = st.selectbox("🍜 Select Cuisine", sorted(df['cuisines'].dropna().unique()))

    filtered = df[
        (df['location'] == location) &
        (df['cuisines'].str.contains(cuisine, na=False))
    ]

    if st.button("✨ Get Recommendations"):

        if filtered.empty:
            st.warning("No results found.")
        else:
            with st.spinner("Analyzing reviews..."):
                results = []
                sentiments = []
                all_reviews = ""

                for _, row in filtered.iterrows():
                    review = str(row['reviews_list'])
                    sentiment = predict_sentiment(review)

                    try:
                        rating = float(str(row['rate']).split('/')[0])
                    except:
                        rating = 0

                    if sentiment == 1:
                        results.append((row['name'], rating))

                    sentiments.append(sentiment)
                    all_reviews += " " + review

                results = sorted(results, key=lambda x: x[1], reverse=True)

            st.success("Top Recommendations")

            cols = st.columns(3)

            for i, (name, rating) in enumerate(results[:6]):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="card">
                        <h3>{name}</h3>
                        <p>⭐ Rating: {rating}</p>
                        <p style="color:#64ffda;">Highly Recommended</p>
                    </div>
                    """, unsafe_allow_html=True)

# ================= TAB 2 =================
with tab2:

    st.markdown('<div class="section">📊 Data Insights</div>', unsafe_allow_html=True)

    if 'filtered' in locals() and not filtered.empty:

        sentiments = []
        all_reviews = ""

        for _, row in filtered.iterrows():
            review = str(row['reviews_list'])
            sentiment = predict_sentiment(review)
            sentiments.append(sentiment)
            all_reviews += " " + review

        pos = sum(sentiments)
        neg = len(sentiments) - pos

        fig, ax = plt.subplots()
        ax.bar(['Positive', 'Negative'], [pos, neg])
        ax.set_title("Sentiment Distribution")
        st.pyplot(fig)

        wc = WordCloud(width=800, height=400, background_color='black').generate(all_reviews)

        fig2, ax2 = plt.subplots()
        ax2.imshow(wc)
        ax2.axis("off")
        st.pyplot(fig2)

    else:
        st.info("Run recommendations first to see insights.")