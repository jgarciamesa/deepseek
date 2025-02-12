#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the Llama 3 8B model and tokenizer from Hugging Face
os.environ['HUGGINGFACE_HUB_CACHE'] = "/scratch/dshah47/.cache/"


# In[2]:


model_name = "/scratch/dshah47/.cache/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/"
# model_name = "/scratch/dshah47/.cache/models--deepseek-ai--DeepSeek-R1-Distill-Llama-70B/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


# ## Loading from /data/datasets

# In[3]:


model_name = "/data/datasets/community/huggingface/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-14B/"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")


# ## Gradio interface

# In[3]:


# Function to generate a response
def chat(input_text):
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=2000, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("</think>")
    print(response[0])  # thinking
    return response[1].lstrip()

# Gradio interface
iface = gr.Interface(fn=chat, 
                     inputs="text", 
                     outputs="text",
                     title="DeepSeek R1 distill Qwen 14B Chat",
                     description="Chat with DeepSeek R1 distill Qwen 14B model from Hugging Face.")

# %%scalene
# Launch the interface
iface.launch(share=True)


# In[ ]:




