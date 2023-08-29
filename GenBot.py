import gradio as gr
import torch
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Initialize the GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)

def clean_output(text):
    # Remove any repeated punctuation
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any spaces before punctuation
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    
    return text.strip()

def generate_response(prompt, temperature, top_p):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, 
                            no_repeat_ngram_size=4, early_stopping=True, 
                            temperature=temperature,  # Use the provided value
                            top_p=top_p)              # Use the provided value
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    response = clean_output(response)
    return response

# Define Gradio interface
def chat_interface(prompt, temperature, top_p):
    return generate_response(prompt, temperature, top_p)

# Gradio components
inputs = [
    gr.inputs.Textbox(lines=7, label="Your Prompt"),
    gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.5, label="Temperature"),
    gr.inputs.Slider(minimum=0.1, maximum=1.0, step=0.1, default=0.9, label="Top P")
]
outputs = gr.outputs.Textbox(label="GPT-2 Response")

# Launch the Gradio interface
gr.Interface(fn=chat_interface, inputs=inputs, outputs=outputs, title="GPT-2 Chatbot", description="Generate responses using GPT-2").launch()
