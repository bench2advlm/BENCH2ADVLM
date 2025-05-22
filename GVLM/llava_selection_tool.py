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

    
def llava_chat_selection(
         llava_parser,
         front_img,
	    advlm_answer: str = None
	):
	tokenizer, model, image_processor, context_len = llava_parser
	LLAVA_PROMPT =f"""You are an expert in autonomous driving command conversion.
	Your task is to transform a high-level driving decision from an autonomous driving model into control signals for the vehicle using predefined values, taking into account both:
	1. the driving decision provided by the driving model, and
	2. the front-view image of the ego-car’s current perspective
	
	Control signal options:
	- "throttle" (controlls the vehicle's acceleration):
	  0.0 (none), 0.3 (light acceleration), 0.6 (moderate acceleration), 0.9 (full acceleration)

	- "brake" (controlls the vehicle's deceleration):
	  0.0 (none), 0.3 (light deceleration), 0.6 (moderate deceleration), 0.9 (full braking)
	
	- "steer" (controlls the vehicle's steering angle):
	  -1.0 (sharp left turn), -0.5 (left turn), -0.2 (slight left turn), 0.0 (straight forward), 0.2 (slight right turn), 0.5 (right turn), 1.0 (sharp right turn)

	**Decision input:** `{advlm_answer}`  
	**Image input** `<image-placeholder>`

	Please follow these instructions strictly:
	1. Choose ONE value per control signal (throttle, steer, brake) from the predefined options.
	2. Provide a clear reasoning process for every signal before selecting its final value. 
	3. Output only a valid JSON string containing exactly these three keys and their float values.
	
	Example valid outputs:
	1. "the car should speed up moderately and turn slightly right":
	{{"throttle": 0.6, "steer": 0.2, "brake": 0.0}}
	
	2. "the car should slow down while turning left":
	{{"throttle": 0.0, "steer": -0.5, "brake": 0.3}}
	
	3. "the car should maintain speed and stay straight":
	{{"throttle": 0.3, "steer": 0.0, "brake": 0.0}}

	4. "the car should come to a stop":
	{{"throttle": 0.0, "steer": 0.0, "brake": 0.9}}
	
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
     
	VALID_VALUES = {
          "throttle": {0.0, 0.3, 0.6, 0.9},
          "steer": {-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0},
          "brake": {0.0, 0.3, 0.6, 0.9}
     }
          
	try:
		llava_control = json.loads(llava_answer)
		for key in ["throttle", "steer", "brake"]:
		     value = llava_control.get(key, 0.0)
		     if value not in VALID_VALUES[key]:
		          closest = min(VALID_VALUES[key], key=lambda x: abs(x - value))
		          llava_control[key] = closest
	except (json.JSONDecodeError, KeyError) as e:
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