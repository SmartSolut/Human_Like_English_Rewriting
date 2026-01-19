"""
Streamlit App for Human-Like English Rewriting System
"""
import os
import sys
import yaml
import torch
import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import time
import re

# Page config
st.set_page_config(
    page_title="Human-Like English Rewriting",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stTextArea textarea {
        min-height: 300px;
    }
    .metric-card {
        background: #f0f4ff;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load config
@st.cache_resource
def load_config():
    try:
        with open("config.yaml", 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return None

config = load_config()

# Model loading
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_path = "./models/final"
        base_model_name = "t5-base"
        
        if not os.path.exists(model_path):
            st.error(f"❌ Model not found at {model_path}")
            return None, None, None
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load base model
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, model_path)
        model.eval()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not torch.cuda.is_available():
            model = model.to(device)
        
        return model, tokenizer, device
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None, None

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
    st.session_state.model = None
    st.session_state.tokenizer = None
    st.session_state.device = None

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    tone = st.selectbox(
        "Tone",
        ["casual", "formal", "academic"],
        index=0
    )
    
    strength = st.selectbox(
        "Strength (قوة إعادة الصياغة)",
        ["light", "medium", "strong", "ultra run"],
        index=1
    )
    
    max_length = st.slider(
        "Max Length",
        min_value=128,
        max_value=512,
        value=512,
        step=64
    )
    
    if st.button("🔄 Load Model", use_container_width=True):
        with st.spinner("Loading model..."):
            model, tokenizer, device = load_model()
            if model is not None:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.device = device
                st.session_state.model_loaded = True
                st.success("✅ Model loaded successfully!")
            else:
                st.error("❌ Failed to load model")
    
    if st.session_state.model_loaded:
        st.success("✅ Model Ready")
        if torch.cuda.is_available():
            st.info(f"🖥️ Device: CUDA")
        else:
            st.info(f"🖥️ Device: CPU")

# Main content
st.markdown("""
<div class="main-header">
    <h1>🔄 Human-Like English Rewriting</h1>
    <p>Rewrite your text naturally while preserving meaning</p>
</div>
""", unsafe_allow_html=True)

# Text input
col1, col2 = st.columns(2)

with col1:
    st.subheader("📝 Input Text")
    input_text = st.text_area(
        "Enter your text here",
        height=400,
        placeholder="Type or paste your text here...",
        key="input"
    )
    input_word_count = len(input_text.split()) if input_text else 0
    st.caption(f"Words: {input_word_count}")

with col2:
    st.subheader("✨ Paraphrased Text")
    
    if st.button("🔄 Rewrite Text", type="primary", use_container_width=True):
        if not st.session_state.model_loaded:
            st.error("❌ Please load the model first from the sidebar!")
        elif not input_text.strip():
            st.warning("⚠️ Please enter some text to rewrite!")
        else:
            with st.spinner("Rewriting text..."):
                start_time = time.time()
                
                try:
                    model = st.session_state.model
                    tokenizer = st.session_state.tokenizer
                    device = st.session_state.device
                    
                    # Prepare input
                    input_prompt = f"humanize: {input_text}"
                    
                    # Tokenize
                    inputs = tokenizer(
                        input_prompt,
                        max_length=max_length,
                        padding=True,
                        truncation=True,
                        return_tensors='pt'
                    ).to(device)
                    
                    # Generation parameters
                    if strength == "light":
                        temperature = 1.15
                        repetition_penalty = 1.5
                        length_penalty = 1.05
                        num_beams = 2
                    elif strength == "strong" or strength == "ultra run":
                        temperature = 1.5
                        repetition_penalty = 1.65
                        length_penalty = 1.3
                        num_beams = 2
                    else:  # medium
                        temperature = 1.35
                        repetition_penalty = 1.55
                        length_penalty = 1.15
                        num_beams = 2
                    
                    # Adjust for tone
                    if tone == "academic":
                        temperature = max(0.5, temperature - 0.12)
                        repetition_penalty = max(1.6, repetition_penalty + 0.08)
                        length_penalty += 0.2
                    elif tone == "formal":
                        temperature = max(0.5, temperature - 0.08)
                        repetition_penalty = max(1.55, repetition_penalty + 0.05)
                        length_penalty += 0.15
                    
                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=max_length,
                            min_length=max(20, int(inputs['input_ids'].shape[1] * 0.8)),
                            num_beams=num_beams,
                            early_stopping=False,
                            repetition_penalty=repetition_penalty,
                            length_penalty=length_penalty,
                            no_repeat_ngram_size=3,
                            do_sample=True,
                            temperature=temperature,
                            top_p=0.93,
                            top_k=42,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    
                    # Decode
                    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean up
                    if rewritten.startswith("humanize:"):
                        rewritten = rewritten.replace("humanize:", "").strip()
                    
                    processing_time = time.time() - start_time
                    
                    # Display result
                    st.text_area(
                        "Result",
                        rewritten,
                        height=400,
                        key="output"
                    )
                    
                    output_word_count = len(rewritten.split())
                    st.caption(f"Words: {output_word_count}")
                    
                    # Metrics
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Processing Time", f"{processing_time:.2f}s")
                    with col_b:
                        st.metric("Input Words", input_word_count)
                    with col_c:
                        st.metric("Output Words", output_word_count)
                    
                    # Copy button
                    st.code(rewritten, language=None)
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
    
    if 'rewritten' in locals():
        st.text_area(
            "Result",
            rewritten,
            height=400,
            key="output_display"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Human-Like English Rewriting System | Powered by T5 + LoRA</p>
</div>
""", unsafe_allow_html=True)
