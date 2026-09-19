import os
from pathlib import Path
import pandas as pd
import nltk
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from sumeval.metrics.rouge import RougeCalculator

# 1. Load Environment Variables (.env)
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

api_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
if api_token:
    login(token=api_token)

# 2. Evaluation Metric Setup
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
rouge = RougeCalculator()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def evaluate_predictions(references, predictions):
    bleu_scores, rouge_scores, meteor_scores = [], [], []
    chencherry = SmoothingFunction().method1
    
    for ref, pred in zip(references, predictions):
        ref_tokens = str(ref).lower().split()
        pred_tokens = str(pred).lower().split()
        
        bleu_scores.append(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=chencherry))
        # Fixed keyword argument 'references' to accept list
        rouge_scores.append(rouge.rouge_n(summary=str(pred), references=[str(ref)], n=1))
        meteor_scores.append(meteor_score([ref_tokens], pred_tokens))
        
    ref_emb = embedding_model.encode(references, convert_to_tensor=True)
    pred_emb = embedding_model.encode(predictions, convert_to_tensor=True)
    cosine_sim = torch.nn.functional.cosine_similarity(ref_emb, pred_emb).mean().item()
    
    return {
        "BLEU": sum(bleu_scores) / len(bleu_scores),
        "ROUGE-1": sum(rouge_scores) / len(rouge_scores),
        "METEOR": sum(meteor_scores) / len(meteor_scores),
        "Cosine Similarity": cosine_sim
    }

def run_model_benchmark(model_id, df):
    print(f"\n--- Running Evaluation for: {model_id} ---")
    
    llm = HuggingFacePipeline.from_model_id(
        model_id=model_id,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 80,
            "temperature": 0.3,
            "do_sample": True
        }
    )
    
    template = """<|im_start|>system
You are an expert e-commerce copywriter. Write a concise, engaging, 1-sentence product description based on the provided metadata.<|im_end|>
<|im_start|>user
Metadata: {metadata}<|im_end|>
<|im_start|>assistant
Description:"""
    
    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm
    
    input_col = "product_metadata"
    ref_col = "product_description"
    
    predictions = []
    for meta in df[input_col]:
        res = chain.invoke({"metadata": meta})
        clean_res = str(res).split("Description:")[-1].strip()
        predictions.append(clean_res)
        
    return evaluate_predictions(df[ref_col].tolist(), predictions)

# 3. Main Execution Block
if __name__ == "__main__":
    df = pd.read_csv("product_data.csv")
    
    models_to_test = [
        "Qwen/Qwen2.5-1.5B-Instruct",
        # "meta-llama/Llama-3.2-1B-Instruct"  # Remove '#' when Meta grants access
    ]
    
    results = {}
    for model in models_to_test:
        results[model] = run_model_benchmark(model, df)
        
    print("\n================ FINAL BENCHMARK RESULTS ================")
    results_df = pd.DataFrame(results).T
    print(results_df.to_string())