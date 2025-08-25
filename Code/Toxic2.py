import streamlit as st
import pickle
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import re

# Set page config
st.set_page_config(
    page_title="Toxic Comment Classifier",
    page_icon="🔍",
    layout="wide"
)

# Define common toxic words
TOXIC_WORDS = {
    'profanity': ['fuck', 'shit', 'ass', 'damn', 'bitch'],
    'hate_speech': ['racist', 'nigger', 'nazi', 'faggot', 'retard'],
    'insults': ['idiot', 'stupid', 'dumb', 'moron', 'loser'],
    'threats': ['kill', 'die', 'murder', 'hurt', 'attack'],
    'harassment': ['stalker', 'creep', 'pervert', 'harassment']
}

# Load the saved model and vectorizer
@st.cache_resource
def load_saved_model():
    model = load_model('toxic_comment_model.h5')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

def predict_toxicity(text, model, vectorizer):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized.toarray())[0][0]
    return prediction

def identify_toxic_words(text):
    text_lower = text.lower()
    found_toxic_words = {}

    for category, words in TOXIC_WORDS.items():
        found_words = []
        for word in words:
            if word in text_lower:
                found_words.append(word)
        if found_words:
            found_toxic_words[category] = found_words

    return found_toxic_words
def create_gauge_chart(probability):
    """
    Creates a gauge chart to display the toxicity probability.
    """
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red" if probability > 0.5 else "green"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            },
            title={'text': "Toxicity Probability (%)"}
        )
    )
    return fig
def create_toxic_words_chart(toxic_words_dict):
    categories = []
    word_counts = []

    for category, words in toxic_words_dict.items():
        categories.append(category)
        word_counts.append(len(words))

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=word_counts,
            marker_color=['red', 'orange', 'yellow', 'purple', 'brown']
        )
    ])

    fig.update_layout(
        title="Toxic Words by Category",
        xaxis_title="Category",
        yaxis_title="Number of Words",
        showlegend=False
    )

    return fig
def create_prediction_history_chart(prediction_history):
    """
    Creates a line chart to visualize prediction probabilities over time.
    """
    fig = px.line(
        prediction_history,
        x=prediction_history.index,
        y="probability",
        title="Prediction History",
        labels={"index": "Prediction Number", "probability": "Toxicity Probability"},
        markers=True
    )
    fig.update_traces(line=dict(color="blue"), marker=dict(size=8))
    fig.update_layout(
        xaxis_title="Prediction Number",
        yaxis_title="Toxicity Probability",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )
    return fig
def main():
    st.title("💬 Toxic Comment Classifier")
    st.write("Enter a comment (up to 200 words) to analyze its toxicity level")

    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = pd.DataFrame(columns=['text', 'probability'])

    try:
        model, vectorizer = load_saved_model()

        col1, col2 = st.columns([2, 1])

        with col1:
            comment = st.text_area("Enter your comment:", height=100, max_chars=1000)

            if st.button("Analyze Comment"):
                if comment:
                    # Check word limit
                    word_count = len(comment.split())
                    if word_count > 200:
                        st.warning("Please limit your text to 200 words. Current word count: " + str(word_count))
                    else:
                        # Get prediction
                        probability = predict_toxicity(comment, model, vectorizer)

                        # Identify toxic words
                        toxic_words = identify_toxic_words(comment)

                        # Add to prediction history
                        new_prediction = pd.DataFrame({
                            'text': [comment],
                            'probability': [probability]
                        })
                        st.session_state.prediction_history = pd.concat(
                            [st.session_state.prediction_history, new_prediction],
                            ignore_index=True
                        )

                        # Display gauge chart
                        st.plotly_chart(create_gauge_chart(probability), use_container_width=True)

                        # Display toxic words analysis
                        st.write("### Toxic Words Analysis:")
                        if toxic_words:
                            st.warning("Found toxic words in the following categories:")
                            for category, words in toxic_words.items():
                                st.write(f"**{category.replace('_', ' ').title()}:** {', '.join(words)}")

                            # Display toxic words chart
                            st.plotly_chart(create_toxic_words_chart(toxic_words), use_container_width=True)
                        else:
                            st.success("No common toxic words found in the text")

                        # Display overall result
                        st.write("### Overall Analysis Result:")
                        if probability > 0.5:
                            st.error(f"This comment is likely toxic (Probability: {probability:.2%})")
                        else:
                            st.success(f"This comment appears non-toxic (Probability: {probability:.2%})")
                else:
                    st.warning("Please enter a comment to analyze")

        with col2:
            st.write("### Statistics")
            if not st.session_state.prediction_history.empty:
                total_comments = len(st.session_state.prediction_history)
                toxic_comments = len(st.session_state.prediction_history[
                    st.session_state.prediction_history['probability'] > 0.5
                ])

                st.metric("Total Comments Analyzed", total_comments)
                st.metric("Toxic Comments Detected", toxic_comments)
                st.metric("Non-toxic Comments", total_comments - toxic_comments)

        if not st.session_state.prediction_history.empty:
            st.write("### Prediction History")
            history_chart = create_prediction_history_chart(
                st.session_state.prediction_history.reset_index()
            )
            st.plotly_chart(history_chart, use_container_width=True)

            st.write("### Recent Predictions")
            st.dataframe(
                st.session_state.prediction_history.tail(5)[['text', 'probability']]
                .style.format({'probability': '{:.2%}'})
            )

            if st.button("Clear History"):
                st.session_state.prediction_history = pd.DataFrame(columns=['text', 'probability'])
                st.experimental_rerun()

    except Exception as e:
        st.error(f"Error loading model or vectorizer: {str(e)}")
        st.write("Please ensure the model and vectorizer files are in the correct location.")

if __name__ == "__main__":
    main()
