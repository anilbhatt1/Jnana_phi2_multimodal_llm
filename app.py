import os
import gc
import json
import torch
import torch.nn as nn
from torch.nn import functional as F
import re
import random
import numpy as np
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import peft
from peft import LoraConfig
from peft import PeftModel
import whisperx
import requests
from io import BytesIO

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))  # Print GPU name
else:
    device = torch.device("cpu")
    print("Using CPU")


model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
bos_token_id = tokenizer.bos_token_id
pad_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.bos_token_id
eoc_string = 'caption image:'
eoc_tokens = tokenizer.encode(eoc_string)
eoq_string = 'end of question:'
eoq_tokens = tokenizer.encode(eoq_string)

model_name = "microsoft/phi-2"
base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  low_cpu_mem_usage=True,
                                                  return_dict=True,
                                                  torch_dtype=torch.float16,
                                                  trust_remote_code=True).to(device)
base_model.resize_token_embeddings(len(tokenizer))

user = "anilbhatt1"  # put your user name here
model_name = "phi2-proj-offset-peft-model"
model_id = f"{user}/{model_name}"

# Merging the peft-model(trained adapters) downloaded from HF with base-phi2-model
merged_phi2 = peft.PeftModel.from_pretrained(base_model, model_id)

vision_model_name = 'openai/clip-vit-base-patch32' ## torch.Size([1, 49, 768])
clip_patches = 49
clip_processor = CLIPImageProcessor.from_pretrained(vision_model_name)
clip_model = CLIPVisionModel.from_pretrained(vision_model_name).to(device)

class ClipProjectionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
class Phi2ProjModel(nn.Module):
    def __init__(self, clip_model, clip_processor, proj_model, phi2_model, clip_embed_dim=768, phi2_dim=2560):
        super(Phi2ProjModel, self).__init__()
        self.clip_embed_dim = clip_embed_dim
        self.phi2_dim = phi2_dim
        self.proj_lin_layer = nn.Linear(clip_embed_dim, phi2_dim)
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.proj_model = proj_model
        self.phi2_model = phi2_model

    def forward(self, input_embed):
        max_len = 100
        output = self.phi2_model.generate(inputs_embeds=input_embed,
                                          max_new_tokens=max_len,
                                          return_dict_in_generate = True,
                                          bos_token_id=bos_token_id,
                                          pad_token_id=bos_token_id,
                                          eos_token_id=bos_token_id)

        return output
    
projection_layer = ClipProjectionBlock(2560).to(device)

phi2_proj_model = Phi2ProjModel(clip_model, clip_processor, projection_layer, merged_phi2).to(device)

phi2_proj_model.proj_lin_layer.load_state_dict(torch.load('./phi2_proj_model_offset_ll.pth'))
phi2_proj_model.proj_model.load_state_dict(torch.load('./phi2_proj_model_offset_projmodel.pth'))

audio_model = whisperx.load_model("small", "cuda", compute_type="float16")

def prepare_input_embed(img=None, audio=None, text=None):

    input_embed_exists = 0

    inputs_given = []

    if img is not None:
        inputs = clip_processor(images=img, return_tensors="pt").to(device)
        clip_output = clip_model(**inputs, output_hidden_states=True)  # B, 50, 768
        clip_embeddings = clip_output.last_hidden_state[:,1:, :]     # B, 49, 768
        image_embed = phi2_proj_model.proj_lin_layer(clip_embeddings)   # B, 49, 2560
        image_embed = phi2_proj_model.proj_model(image_embed)    # B, 49, 2560
        B, _, C = image_embed.shape

        eoc_tkn_tensor = torch.tensor(eoc_tokens, dtype=torch.int64).to(device)  # [4] -> EOI token matrix
        eoc_tensor = eoc_tkn_tensor.repeat(B, 1)  # [B, 4]
        eoc_embed = phi2_proj_model.phi2_model.base_model.model.model.embed_tokens(eoc_tensor)  # B, 4, 2560 -> EOI embeddings (torch.float32)

        input_image_embed  = torch.cat([image_embed, eoc_embed], dim=1)  #B, 53, 2560 -> Adding EOI embeddings to indicate end of image
        input_image_embed = input_image_embed.to(dtype=torch.float16)

    if audio is not None:
        audio_tkn_tensor = torch.tensor(audio, dtype=torch.int64).to(device)  # [4] -> EOI token matrix
        audio_tkn_tensor = audio_tkn_tensor.unsqueeze(0)
        audio_embed = phi2_proj_model.phi2_model.base_model.model.model.embed_tokens(audio_tkn_tensor)        

    if text is not None:
        text_tkn_tensor = torch.tensor(text, dtype=torch.int64).to(device)  # [4] -> EOI token matrix
        text_tkn_tensor = text_tkn_tensor.unsqueeze(0)
        text_embed = phi2_proj_model.phi2_model.base_model.model.model.embed_tokens(text_tkn_tensor)

    # If image is present, it gets 1st place in input_embed
    if img is not None:
        input_embed = input_image_embed
        input_embed_exists = 1

    if audio is not None:
        # If input_embed is already present, that means image was present. So, append audio_embed to it
        if input_embed_exists:
            input_embed = torch.cat([input_embed, audio_embed], dim=1)
        # If input_embed is not there, that means image is not there. So, give audio_embed as input_embed
        else:
            input_embed = audio_embed
            input_embed_exists = 1
        inputs_given.append(audio)

    if text:
        # If input_embed is already present, that means image/audio are present. So, append text_embed to it
        if input_embed_exists:
            if audio is not None:
                input_embed = torch.cat([input_embed, text_embed], dim=1)              
            else:
                input_embed = torch.cat([input_embed, text_embed], dim=1)
        # If input_embed is not there, that means neither image not audio there. So, give text_embed as input_embed
        else:
            input_embed = text_embed
            input_embed_exists = 1
        inputs_given.append(text)
    
    inputs_given.append(eoq_tokens)

    eoq_tkn_tensor = torch.tensor(eoq_tokens, dtype=torch.int64).to(device)  # [4] -> EOI token matrix
    B = 1
    eoq_tensor = eoq_tkn_tensor.repeat(B, 1)  # [B, 4]
    eoq_embed = phi2_proj_model.phi2_model.base_model.model.model.embed_tokens(eoq_tensor)  # B, 4, 2560 -> EOI embeddings (torch.float32)
    input_embed  = torch.cat([input_embed, eoq_embed], dim=1)

    return input_embed

def gradio_get_answers_fn(image=None, audio=None, text=None):
    audio_tokens = None
    text_tokens = None
    if audio:
        audio_result = audio_model.transcribe(audio)
        audio_text = ''
        for seg in audio_result['segments']:
            audio_text += seg['text']
        audio_text = audio_text.strip()
        audio_tokens = tokenizer.encode(audio_text)

    if text:
        text_tokens = tokenizer.encode(text)

    if image or audio or text:
        input_embed = prepare_input_embed(image, audio_tokens, text_tokens)
        with torch.no_grad():
            output = phi2_proj_model(input_embed)
            out_text = tokenizer.batch_decode(output.sequences[:, 1:])[0] 
            out_text = out_text.replace("<|endoftext|>", "")  
    else:
        out_text = "I didn't get any input. Give me an image/audio/text or combination of these 3 and get the answer back !"

    return out_text

import gradio as gr

markdown_description = """
- Jñāna is a Multimodal LLM app that can accept input as image, text or audio
- Based on the input you can query the app for more details
- Uses **microsoft/phi-2 qlora** optimized model finetuned on **instruct150k** dataset
- Uses **whisperX** model for audio
"""
demo = gr.Interface(fn=gradio_get_answers_fn,
                    inputs=[
                            gr.Image(type="pil", label="Image"),
                            gr.Audio(label="Audio Query", sources=['microphone', 'upload'], type='filepath'),
                            gr.Textbox(info="How may I help you ? please enter your prompt here...", label="Text Query")
                           ],
                    outputs=gr.Textbox(label="Response"),
                    title="Jñāna - Phi2 Multiomodal Conversation Agent",
                    description=markdown_description,
                    article=" **Credits** : https://theschoolof.ai/ || https://github.com/mshumer/gpt-llm-trainer || https://github.com/huggingface/peft/tree/main/examples/multilayer_perceptron ")

demo.queue().launch(share=True)