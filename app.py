
import streamlit as st
import numpy as np
import pandas as pd
from apputil import *


st.write(
'''
# Week 6: Markov Chain Text Generation

This app demonstrates text generation using Markov chains. The generator learns patterns from 
inspirational quotes and creates new text based on word transition probabilities.
''')

# Initialize the corpus and text generator
@st.cache_data
def load_corpus():
    """Cache the corpus loading to avoid repeated downloads."""
    return fetch_and_clean_quotes()

@st.cache_data
def create_generator(corpus):
    """Cache the text generator creation."""
    text_gen = MarkovText(corpus)
    text_gen.get_term_dict()
    return text_gen

# Load data
with st.spinner('Loading inspirational quotes corpus...'):
    corpus = load_corpus()
    text_gen = create_generator(corpus)

# Display corpus info
st.write("---")
st.subheader("📚 Corpus Information")
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Words", len(corpus.split()))
with col2:
    st.metric("Unique Tokens", len(text_gen.term_dict))

# Show sample of corpus
with st.expander("View Corpus Sample"):
    st.text(corpus[:500] + "...")

# Show term dictionary sample
with st.expander("View Term Dictionary Sample"):
    sample_items = list(text_gen.term_dict.items())[:10]
    dict_df = pd.DataFrame([
        {"Token": key, "Following Words": ", ".join(values[:5]) + ("..." if len(values) > 5 else "")}
        for key, values in sample_items
    ])
    st.dataframe(dict_df, use_container_width=True)

st.write("---")
st.subheader("🎲 Text Generation")

# Input controls
col1, col2 = st.columns(2)

with col1:
    # Number of words to generate
    word_count = st.number_input("Number of words to generate:", 
                                 value=15, 
                                 min_value=5,
                                 max_value=100,
                                 step=1, 
                                 format="%d")

with col2:
    # Seed term selection
    available_terms = ["(Random)"] + sorted(list(text_gen.term_dict.keys()))
    seed_selection = st.selectbox("Starting word (seed term):", 
                                  available_terms,
                                  index=0)

# Generate button
if st.button("🎯 Generate Text", type="primary"):
    try:
        # Determine seed term
        seed_term = None if seed_selection == "(Random)" else seed_selection
        
        # Generate text
        generated_text = text_gen.generate(seed_term=seed_term, term_count=word_count)
        
        if generated_text:
            # Display generated text
            st.success("Generated Text:")
            st.write(f"*\"{generated_text}\"*")
            
            # Show statistics
            st.caption(f"Generated {len(generated_text.split())} words")
            if seed_term:
                st.caption(f"Started with seed term: '{seed_term}'")
        else:
            st.error("Failed to generate text. Please try again.")
            
    except ValueError as e:
        st.error(f"Error: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

st.write("---")

# Multiple generations comparison
st.subheader("🔄 Multiple Generations Comparison")
st.write("Generate multiple texts with the same settings to see the randomness in action:")

col1, col2 = st.columns(2)
with col1:
    comparison_count = st.number_input("Number of variations:", 
                                      value=3, 
                                      min_value=2,
                                      max_value=10,
                                      step=1,
                                      format="%d")
with col2:
    comparison_seed = st.selectbox("Comparison seed term:", 
                                   available_terms,
                                   index=0,
                                   key="comparison_seed")

if st.button("🎲 Generate Variations"):
    seed_term = None if comparison_seed == "(Random)" else comparison_seed
    
    st.write(f"**{comparison_count} variations with {word_count} words each:**")
    
    for i in range(comparison_count):
        try:
            generated = text_gen.generate(seed_term=seed_term, term_count=word_count)
            if generated:
                st.write(f"{i+1}. *\"{generated}\"*")
        except Exception as e:
            st.write(f"{i+1}. Error: {e}")

st.write("---")

# Educational section
with st.expander("📖 How Markov Chain Text Generation Works"):
    st.write("""
    **Markov chains** are probabilistic models where the next state depends only on the current state,
    not on the sequence of events that preceded it (memoryless property).
    
    **In text generation:**
    1. We analyze a corpus to learn which words typically follow other words
    2. We build a transition dictionary mapping each word to its possible successors
    3. We include duplicates to preserve natural probability distributions
    4. During generation, we randomly select the next word from the possible successors
    5. This process continues for the desired number of words
    
    **Why include duplicates?**
    If "is" is followed by "the" three times and "a" once in our corpus, keeping duplicates 
    means "the" has a 75% chance and "a" has a 25% chance of being selected - preserving 
    the natural frequency patterns from the original text.
    """)

# Footer
st.write("---")
st.caption("Built with Streamlit • Markov Chain Text Generation Exercise")