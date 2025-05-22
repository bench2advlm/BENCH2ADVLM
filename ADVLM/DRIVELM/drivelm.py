import cv2
import torch
from PIL import Image
import json
import argparse
import torchvision.transforms as transforms

import DRIVELM.llama_drivelm as llama_drivelm

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def load_drivelm():
	llama_dir = "ADVLM/DRIVELM/ckpts/llama_model_weights"
	checkpoint = "ADVLM/DRIVELM/ckpts/checkpoint-7.pth"
	model, preprocess = llama_drivelm.load(checkpoint, llama_dir, llama_type="7B")
	return model, preprocess


def process_image(input_dict, transform= transforms.Compose([
	   transforms.Resize((224, 224), interpolation=BICUBIC),
	   transforms.ToTensor(),
	   transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])])):
	image = input_dict['img'][0]
	if transform:
		image = transform(image)
		image = image.unsqueeze(0)
	return image


def get_drivelm_answer(input_dict, model):
	image = process_image(input_dict)
	question ="<image>\nBased on the current scene, what action should the vehicle take next?" 
	prompt = llama_drivelm.format_prompt(question)
	prompts = [prompt]
	images = image.unsqueeze(0).to('cuda')
	# inference
	results = model.generate(images, prompts, temperature=0.2, top_p=0.1)
	return results[0]
