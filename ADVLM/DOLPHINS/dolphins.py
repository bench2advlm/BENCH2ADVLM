import os
from typing import Union
from PIL import Image
import mimetypes
import cv2
import torch
import sys
from huggingface_hub import hf_hub_download
from peft import (
    get_peft_model,
    LoraConfig,
    get_peft_model_state_dict,
    PeftConfig,
    PeftModel
)

from DOLPHINS.configs.lora_config import openflamingo_tuning_config
from DOLPHINS.mllm.src.factory import create_model_and_transforms

generation_kwargs = {'max_new_tokens': 512, 'temperature': 1,
                        'top_k': 0, 'top_p': 1, 'no_repeat_ngram_size': 3, 'length_penalty': 1,
                        'do_sample': False,
                        'early_stopping': True}


def load_pretrained_modoel():
    peft_config, peft_model_id = None, None
    peft_config = LoraConfig(**openflamingo_tuning_config)
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14-336",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path="anas-awadalla/mpt-7b", # anas-awadalla/mpt-7b
        tokenizer_path="anas-awadalla/mpt-7b",  # anas-awadalla/mpt-7b
        cross_attn_every_n_layers=4,
        use_peft=True,
        peft_config=peft_config,
    )

    checkpoint_path = hf_hub_download("gray311/Dolphins", "checkpoint.pt")
    model.load_state_dict(torch.load(checkpoint_path), strict=False)
    model.half().cuda()

    return model, image_processor, tokenizer


def load_dolphins():
	model, image_processor, tokenizer = load_pretrained_modoel()
	tokenizer.eos_token_id = 50277
	tokenizer.pad_token_id = 50277
	return model, image_processor, tokenizer


def get_dolphins_answer(input_dict, model, image_processor, tokenizer):
	imgs = input_dict['front_imgs']
	speeds = input_dict['speeds']
	speeds_str = ', '.join(map(str, speeds))
	if isinstance(imgs, list):
		pass
	elif isinstance(imgs, Image.Image):
		imgs = [imgs]
	else:
		raise ValueError("Input should be a list of images or a single image.")
	vision_x = torch.stack([image_processor(image) for image in imgs], dim=0).unsqueeze(0).unsqueeze(0)
	assert vision_x.shape[2] == len(imgs)
    
	instruction = "What should the ego-car do in the future? Please describe the next driving action in detail."
	prompt = [
		f"USER: <image> is a driving video. {instruction} GPT:<answer>"
	]

	inputs = tokenizer(prompt, return_tensors="pt", ).to(model.device)
	
	# inference
	generated_tokens = model.generate(
		vision_x=vision_x.half().cuda(),
		lang_x=inputs["input_ids"].cuda(),
		attention_mask=inputs["attention_mask"].cuda(),
		num_beams=3,
		**generation_kwargs,
	)
	
	# post-process
	generated_tokens = generated_tokens.cpu().numpy()
	if isinstance(generated_tokens, tuple):
		generated_tokens = generated_tokens[0]
	inference_text = tokenizer.batch_decode(generated_tokens)
	last_answer_index = inference_text[0].rfind("<answer>")
	content_after_last_answer = inference_text[0][last_answer_index + len("<answer>"):]
	final_answer = content_after_last_answer[:content_after_last_answer.rfind("<|endofchunk|>")]
	
	return final_answer