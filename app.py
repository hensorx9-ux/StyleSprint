import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

# 1. Page Config
st.set_page_config(
    page_title="StyleSprint - E-Commerce AI Copywriter",
    page_icon="🛍️",
    layout="centered"
)

# 2. Environment & Token Setup
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if api_token:
    login(token=api_token)

# 3. Model Pipeline Loader (Cached in Memory)
@st.cache_resource
def load_llm_pipeline():
    llm = HuggingFacePipeline.from_model_id(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 80,
            "temperature": 0.3,
            "do_sample": True,
            "clean_up_tokenization_spaces": False
        }
    )
    
    few_shot_template = """<|im_start|>system
You are an expert e-commerce copywriter. Write a concise, engaging, 1-sentence product description based on the provided metadata.

Example 1:
Metadata: Running Shoes, Lightweight Mesh, Men's Size 10, Red
Description: Lightweight red running shoes featuring breathable mesh for maximum comfort during intense workouts.

Example 2:
Metadata: Leather Tote Bag, Vintage Brown, Zipper Closure
Description: A stylish vintage brown leather tote bag crafted with a secure zipper closure for everyday convenience.<|im_end|>
<|im_start|>user
Metadata: {metadata}<|im_end|>
<|im_start|>assistant
Description:"""

    prompt = PromptTemplate.from_template(few_shot_template)
    return prompt | llm

# Load Model
with st.spinner("Loading AI model into RAM..."):
    chain = load_llm_pipeline()

# 4. Interface Layout
st.title("🛍️ StyleSprint Product Generator")
st.markdown("Generate optimized, high-converting product descriptions directly from metadata using local instruct models.")

st.divider()

with st.form("generator_form"):
    metadata_input = st.text_area(
        "Product Metadata",
        placeholder="e.g., Wireless Noise-Canceling Headphones, Matte Black, 30-Hour Battery Life",
        height=100
    )
    submit_button = st.form_submit_button("Generate Description", use_container_width=True)

# 5. Inference Logic
if submit_button:
    if not metadata_input.strip():
        st.warning("Please enter metadata before generating.")
    else:
        with st.spinner("Writing description..."):
            res = chain.invoke({"metadata": metadata_input})
            clean_description = str(res).split("Description:")[-1].strip()
            
            st.subheader("Generated Description")
            st.success(clean_description)