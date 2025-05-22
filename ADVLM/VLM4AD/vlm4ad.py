import argparse
import torch
import os
from VLM4AD.modules.multi_frame_dataset import MultiFrameDataset
from VLM4AD.modules.multi_frame_model import DriveVLMT5
from transformers import T5Tokenizer
from torchvision import transforms
import json
from torchvision.io import read_image


def load_vlm4ad():
    config = params()
    # Load processors and models
    model = DriveVLMT5(config)

    if config.lm == 'T5-Base':
        processor = T5Tokenizer.from_pretrained('google-t5/t5-base')
    else:
        processor = T5Tokenizer.from_pretrained('google-t5/t5-large')

    processor.add_tokens('<')

    model.load_state_dict(
        torch.load(os.path.join('ADVLM/VLM4AD/multi_frame_results', config.model_name,
                                'latest_model.pth')))
    model = model.float()
    return model, processor


def params():
    defaults = {
        'batch_size': 1,
        'epochs': 15,
        'gpa_hidden_size': 128,
        'freeze_lm': False,
        'lm': 'T5-Base',
        'lora': False,
        'lora_dim': 64,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'max_len': 512,
        'num_workers': 0,
        'model_name': 'T5-Medium'
    }
    return argparse.Namespace(**defaults)
	

def get_vlm4ad_answer(imgs, processor, model, transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
        ])):
    with torch.no_grad():  
        question = "Based on the current scene, what action should the vehicle take next?"
        q_text = f"Question: {question} Answer:"
        imgs = [transform(img.float()).to('cuda') for img in imgs]
        imgs = torch.stack(imgs, dim=0).unsqueeze(0)
        encodings = processor([q_text], padding=True, return_tensors="pt").input_ids.to('cuda')
        # inference
        outputs = model.generate(encodings, imgs)
        text_output = processor.decode(outputs[0], skip_special_tokens=True)

        return text_output
    