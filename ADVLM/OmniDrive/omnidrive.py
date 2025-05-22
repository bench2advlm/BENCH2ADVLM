import argparse
import torch
import mmcv
import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from OmniDrive.projects.mmdet3d_plugin.datasets.pipelines import Compose

def load_omnidrive():
	config_path='ADVLM/OmniDrive/projects/configs/OmniDrive/mask_eva_lane_det_vlm.py',
	checkpoint_path='ADVLM/OmniDrive/ckpts/iter_10548.pth'
	
	cfg = Config.fromfile(config_path)
	cfg.model.train_cfg = None
	model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
	load_checkpoint(model, checkpoint_path, map_location='cpu')
	model = MMDataParallel(model, device_ids=[0]).eval()
	test_pipeline = Compose(cfg.test_pipeline)
	return model, test_pipeline


def get_omnidrive_answer(input_dict, model, pipeline):
	question = "What should be your next action and why"
	input_dict['vlm_labels']= [question]
	data = pipeline(input_dict)
	data['img'] = [torch.from_numpy(img).cuda().float() for img in data['img']]
	data['input_ids'] = torch.from_numpy(data['input_ids']).cuda().long()
	scatter_keys = ['img', 'input_ids']
	for key in scatter_keys:
		if isinstance(data[key], list):
			data[key] = [d.unsqueeze(0) for d in data[key]]
		else:
			data[key] = data[key].unsqueeze(0)
			
	with torch.no_grad():
		results = model(return_loss=False, rescale=True, **data)
	  
	text_results = results['text_out'][0]
	answer = text_results['A'][0]
	return answer
