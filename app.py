from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Load the pretrained model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Create the FastAPI app
app = FastAPI()

# Define the request body structure
class GenerationInput(BaseModel):
    prompt: str

# Define the generation endpoint
@app.post("/generate")
def generate_text(input: GenerationInput):
    try:
        # Generate text based on input prompt
        generated_text = generator(input.prompt, max_length=150)
        return {"generated_text": generated_text}
    except Exception:
        raise HTTPException(status_code=500, detail="Model failed to generate text")