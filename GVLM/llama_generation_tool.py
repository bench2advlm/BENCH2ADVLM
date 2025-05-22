from typing import List, Optional
from llama import Dialog
import json

LLAMA_SYSTEM ="""You are an expert in autonomous driving command conversion.
Your task is to convert a given high-level driving decision of an autonomous driving vision-language model into precise control signals for the autonomous vehicle.
The vehicle's control signals consist of three components, each represented by a single float value:
- "throttle": a value between 0.0 and 1.0 that controls the vehicle's acceleration (0.0 means no acceleration, 1.0 means full throttle).
- "steer": a value between -1.0 and 1.0 that controls the vehicle's steering angle (negative values indicate left turns, positive values indicate right turns, and 0.0 means straight).
- "brake": a value between 0.0 and 1.0 that controls the vehicle's braking (0.0 means no braking, 1.0 means full braking).
"""


def llama_chat_generation(
         generator,
	    advlm_answer: str = None,
	    temperature: float = 0.6,
	    top_p: float = 0.9,
	    max_gen_len: Optional[int] = None,
	):	
	LLAMA_INSTRUCTION = f"""The driving decision from the autonomous driving VLM is: [{advlm_answer}]. 
	Convert it into control signals for the autonomous vehicle. 
	
	Please follow these instructions strictly:
	1. Read the provided driving decision carefully.
	2. Analyze the decision and determine the appropriate values for throttle, steer, and brake.
	3. Ensure that the output is a valid JSON string containing only the three keys with their respective float values.
	4. Do not include any additional text or explanation outside of the JSON output.
	
	For example, if the decision indicates "accelerate moderately and turn slightly right", an acceptable output might be:
	{{"throttle": 0.5, "steer": 0.2, "brake": 0.0}}
	
	Now, convert the driving decision into control signals in JSON format:"""
	
	dialogs: List[Dialog] = [
	   [
	       {
	           "role": "system",
	           "content": LLAMA_SYSTEM
	       },
	       {
	           "role": "user", 
	           "content": LLAMA_INSTRUCTION
	       },
	   ],
	]
	results = generator.chat_completion(
		dialogs,
		max_gen_len=max_gen_len,
		temperature=temperature,
		top_p=top_p,
	)
	
	for dialog, result in zip(dialogs, results):	
		print(
	       f"> LLAMA {result['generation']['role'].capitalize()}: {result['generation']['content']}"
	     )
	llama_answer = results[0]['generation']['content']
	
	try:
		llama_control = json.loads(llama_answer)
	except json.JSONDecodeError as e:
		print("Error parsing llama_answer:", e)
		llama_control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}
	        
	return llama_control