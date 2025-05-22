import sys
import torch
import json
from PIL import Image
import re

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)


def llava_chat_generation(
         llava_parser,
         front_img,
	    advlm_answer: str = None
	):
	tokenizer, model, image_processor, context_len = llava_parser
	LLAVA_PROMPT =f"""You are an expert in autonomous driving command conversion.
	Your task is to transform a high-level driving decision from an autonomous driving model into precise control signals for the vehicle, taking into account both:
	1. the driving decision provided by the driving model, and
	2. the front-view image of the ego-car’s current perspective
	
	The vehicle control signals consist of three floats:
	- **throttle**: a value between 0.0 and 1.0, controlling the vehicle's acceleration.
	The value indicates accelaration intensity: lower values maintain or gently increase speed; higher values increase speed.
	- **brake**: a value between 0.0 and 1.0, controlling the vehicle's deceleration. 
	The value indicates braking intensity: lower values correspond to lighter braking for gentle deceleration; higher values correspond to stronger braking to reduce speed or stop the vehicle.
	- **steer**: a value between -1.0 and 1.0, controlling the vehicle's steering angle. 
	Sign indicates direction: negative for left turns, positive for right turns. Absolute value determines turn intensity.
		
	**Decision input:** `{advlm_answer}`  
	**Image input** `<image-placeholder>`

	Please follow these instructions strictly:
	
	1. Provide a clear reasoning process for every signal (throttle, steer, brake) before selecting its final value. 
	2. Note that throttle and brake are mutually exclusive (both cannot be >0 simultaneously)
	3. Output only a valid JSON string containing exactly these three keys and their float values.
	
	For example, if the decision indicates "accelerate moderately and maintain the current lane", an acceptable output might be:{{"throttle": 0.5, "steer": 0.0, "brake": 0.0}}.
	
	Now, based on the given decision and front-view image, generate the final JSON:
	"""
	prompt = generate_prompt(LLAVA_PROMPT)
	
	images = [front_img]
	image_sizes = [x.size for x in images]
	images_tensor = process_images(
	   images,
	   image_processor,
	   model.config
	).to(model.device, dtype=torch.float16)
	
	input_ids = (
	   tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
	   .unsqueeze(0)
	   .cuda()
	)
	
	temperature = 0
	top_p = None
	num_beams = 1
	max_new_tokens = 512
	
	with torch.inference_mode():
	   output_ids = model.generate(
	       input_ids,
	       images=images_tensor,
	       image_sizes=image_sizes,
	       do_sample=True if temperature > 0 else False,
	       temperature=temperature,
	       top_p=top_p,
	       num_beams=num_beams,
	       max_new_tokens=max_new_tokens,
	       use_cache=True,
	   )
	
	llava_answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
	print(f"> LLAVA result:{llava_answer}")
	
	start_idx = llava_answer.find('{')
	end_idx = llava_answer.find('}')
	if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
          llava_answer = llava_answer[start_idx:end_idx+1]
          llava_answer = llava_answer.replace("'", '"')  
	try:
		llava_control = json.loads(llava_answer)
		for key in ["throttle", "steer", "brake"]:
		     value = llava_control.get(key, 0.0)
		     llava_control[key] = value
	except json.JSONDecodeError as e:
		print("Error parsing llava_answer:", e)
		llava_control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}
	        
	return llava_control
	
    

def generate_prompt(query):
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    model_name = "llava-v1.5-13b"
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    return prompt