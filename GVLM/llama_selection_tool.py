from typing import List, Optional
from llama import Dialog
import json

LLAMA_SYSTEM = """You are an expert in autonomous driving command conversion.
Your task is to convert a given high-level driving decision into discrete control signals using predefined values.

Control signal options:
- "throttle" (acceleration):
  0.0 (none), 0.3 (low), 0.6 (moderate), 0.9 (high)

- "steer" (steering):
  -1.0 (sharp left), -0.5 (left), -0.2 (slight left),
  0.0 (straight),
  0.2 (slight right), 0.5 (right), 1.0 (sharp right)

- "brake" (braking):
  0.0 (none), 0.3 (light), 0.6 (moderate), 0.9 (hard)

Follow these rules:
1. Choose ONE value per control signal from the predefined options
2. Combine throttle/brake appropriately (both cannot be >0)
3. Output ONLY valid JSON with float values
4. Do not include any additional text or explanation outside of the JSON output
"""


def llama_chat_selection(
        generator,
        advlm_answer: str = None,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
    ):

    LLAMA_INSTRUCTION = f"""Convert this driving decision into control signals: [{advlm_answer}]

Example valid outputs:
1. "speed up moderately and turn slightly right":
{{"throttle": 0.6, "steer": 0.2, "brake": 0.0}}

2. "slow down while turning left":
{{"throttle": 0.0, "steer": -0.5, "brake": 0.3}}

3. "maintain speed and stay straight":
{{"throttle": 0.3, "steer": 0.0, "brake": 0.0}}

Now generate the JSON output for the driving decision:"""

    dialogs: List[Dialog] = [[
        {"role": "system", "content": LLAMA_SYSTEM},
        {"role": "user", "content": LLAMA_INSTRUCTION},
    ]]
    
    results = generator.chat_completion(
        dialogs,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):    
        print(f"> LLAMA {result['generation']['role'].capitalize()}: {result['generation']['content']}")
    llama_answer = results[0]['generation']['content']

    VALID_VALUES = {
        "throttle": {0.0, 0.3, 0.6, 0.9},
        "steer": {-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0},
        "brake": {0.0, 0.3, 0.6, 0.9}
    }
    
    try:
        llama_control = json.loads(llama_answer)
        for key in ["throttle", "steer", "brake"]:
            value = llama_control.get(key, 0.0)
            if value not in VALID_VALUES[key]:
                closest = min(VALID_VALUES[key], key=lambda x: abs(x - value))
                llama_control[key] = closest
    except (json.JSONDecodeError, KeyError) as e:
        print("Error parsing llama_answer:", e)
        llama_control = {"throttle": 0.0, "steer": 0.0, "brake": 0.0}
        
    return llama_control