import os
import json
import datetime
import pathlib
import time
import cv2
import carla
import math

import torch
import carla
import numpy as np
from PIL import Image
from torchvision import transforms as T

from leaderboard.autoagents import autonomous_agent
from team_code.planner import RoutePlanner
from scipy.optimize import fsolve

from OmniDrive.omnidrive import load_omnidrive, get_omnidrive_answer

from llama_generation_tool import llama_chat_generation
from llama_selection_tool import llama_chat_selection
from llava_generation_tool import llava_chat_generation
from llava_selection_tool import llava_chat_selection


SAVE_PATH = os.environ.get('SAVE_PATH', None)
IS_BENCH2DRIVE = os.environ.get('IS_BENCH2DRIVE', None)
PARSER_TYPE = os.environ.get('PARSER_TYPE', None)
SIGNAL_TYPE = os.environ.get('SIGNAL_TYPE', None)
print('*'*10)
print("omnidrive-agent")
print('*'*10)
print(PARSER_TYPE)
print('*'*10)
print(SIGNAL_TYPE)
print('*'*10)


def get_entry_point():
    return 'OMNIDRIVEAgent'


class OMNIDRIVEAgent(autonomous_agent.AutonomousAgent):
	def setup(self, save_name, parser):
		self.parser = parser	
		if IS_BENCH2DRIVE:
			self.save_name = save_name
		else:
			self.save_name = '_'.join(map(lambda x: '%02d' % x, (now.month, now.day, now.hour, now.minute, now.second)))
		
		self.step = -1
		self.wall_start = time.time()
		self.initialized = False

		# omnidrive
		self.model, self.test_pipeline = load_omnidrive()
		self.model.cuda()
		self.model.eval()

		if SAVE_PATH is not None:
			self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / self.save_name
			self.save_path.mkdir(parents=True, exist_ok=False)
			(self.save_path / 'rgb_front').mkdir()
			(self.save_path / 'meta').mkdir()


	def _init(self):
		try:
			locx, locy = self._global_plan_world_coord[0][0].location.x, self._global_plan_world_coord[0][0].location.y
			lon, lat = self._global_plan[0][0]['lon'], self._global_plan[0][0]['lat']
			EARTH_RADIUS_EQUA = 6378137.0
			def equations(vars):
				x, y = vars
				eq1 = lon * math.cos(x * math.pi / 180) - (locx * x * 180) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(x * math.pi / 180) * y
				eq2 = math.log(math.tan((lat + 90) * math.pi / 360)) * EARTH_RADIUS_EQUA * math.cos(x * math.pi / 180) + locy - math.cos(x * math.pi / 180) * EARTH_RADIUS_EQUA * math.log(math.tan((90 + x) * math.pi / 360))
				return [eq1, eq2]
			initial_guess = [0, 0]
			solution = fsolve(equations, initial_guess)
			self.lat_ref, self.lon_ref = solution[0], solution[1]
		except Exception as e:
			print(e, flush=True)
			self.lat_ref, self.lon_ref = 0, 0 
		print(self.lat_ref, self.lon_ref, self.save_name)
		self._route_planner = RoutePlanner(4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref)
		self._route_planner.set_route(self._global_plan, True)
		self.initialized = True
		self.metric_info = {}


	def sensors(self):
		sensors =[
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.80, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': 0.27, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 55.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_FRONT_RIGHT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -2.0, 'y': 0.0, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0,
                    'width': 1600, 'height': 900, 'fov': 110,
                    'id': 'CAM_BACK'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': -0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': -110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_LEFT'
                },
                {
                    'type': 'sensor.camera.rgb',
                    'x': -0.32, 'y': 0.55, 'z': 1.60,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 110.0,
                    'width': 1600, 'height': 900, 'fov': 70,
                    'id': 'CAM_BACK_RIGHT'
                },
                {
                    'type': 'sensor.other.imu',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.05,
                    'id': 'IMU'
                },
                {
                    'type': 'sensor.other.gnss',
                    'x': -1.4, 'y': 0.0, 'z': 0.0,
                    'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                    'sensor_tick': 0.01,
                    'id': 'GPS'
                },
                {
                    'type': 'sensor.speedometer',
                    'reading_frequency': 20,
                    'id': 'SPEED'
                },
                
            ]
        
		if IS_BENCH2DRIVE:
			sensors += [
                    {	
                        'type': 'sensor.camera.rgb',
                        'x': 0.0, 'y': 0.0, 'z': 50.0,
                        'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                        'width': 512, 'height': 512, 'fov': 5 * 10.0,
                        'id': 'bev'
                    }]
		return sensors


	def tick(self, input_data):
		self.step += 1
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20]
		imgs = {}
		for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
			img = cv2.cvtColor(input_data[cam][1][:, :, :3], cv2.COLOR_BGR2RGB)
			_, img = cv2.imencode('.jpg', img, encode_param)
			img = cv2.imdecode(img, cv2.IMREAD_COLOR)
			imgs[cam] = img
		bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
		gps = input_data['GPS'][1][:2]
		speed = input_data['SPEED'][1]['speed']
		compass = input_data['IMU'][1][-1]
		acceleration = input_data['IMU'][1][:3]
		angular_velocity = input_data['IMU'][1][3:6]
		pos = self.gps_to_location(gps)
		near_node, near_command = self._route_planner.run_step(pos)
		if (math.isnan(compass) == True): #It can happen that the compass sends nan for a few frames
			compass = 0.0
			acceleration = np.zeros(3)
			angular_velocity = np.zeros(3)
		
		result = {
		      'imgs': imgs,
		      'gps': gps,
		      'pos':pos,
		      'speed': speed,
		      'compass': compass,
		      'bev': bev,
		      'acceleration':acceleration,
		      'angular_velocity':angular_velocity,
		      'command_near':near_command,
		      'command_near_xy':near_node
		      }
		
		return result

	
	@torch.no_grad()
	def run_step(self, input_data, timestamp):
		if not self.initialized:
			self._init()
		tick_data = self.tick(input_data)
		results = {}
		results['img'] = []
		results['lidar2img'] = []
		results['intrinsics'] = []
		results['timestamp'] = self.step / 10
		for cam in ['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT']:
			results['lidar2img'].append(self.lidar2img[cam])
			results['intrinsics'].append(np.matmul(self.lidar2img[cam], np.linalg.inv(self.lidar2cam[cam])))
			results['img'].append(tick_data['imgs'][cam])
		results['lidar2img'] = np.stack(results['lidar2img'],axis=0)
		raw_theta = tick_data['compass'] if not np.isnan(tick_data['compass']) else 0
		ego_theta = -raw_theta + np.pi/2
		rotation = list(Quaternion(axis=[0, 0, 1], radians=ego_theta))
		can_bus = np.zeros(18)
		can_bus[0] = tick_data['pos'][0]
		can_bus[1] = -tick_data['pos'][1]
		can_bus[3:7] = rotation
		can_bus[7] = tick_data['speed']
		can_bus[10:13] = tick_data['acceleration']
		can_bus[11] *= -1
		can_bus[13:16] = -tick_data['angular_velocity']
		can_bus[16] = ego_theta
		can_bus[17] = ego_theta / np.pi * 180 
		results['can_bus'] = can_bus
		results['command'] = command2nohot(tick_data['command_near'])
		ego2world = np.eye(4)
		ego2world[0:3,0:3] = Quaternion(axis=[0, 0, 1], radians=ego_theta).rotation_matrix
		ego2world[0:2,3] = can_bus[0:2]
		ego_pose = ego2world
		ego_pose_inv = invert_matrix_egopose_numpy(ego_pose)
		results['ego_pose'] = ego_pose
		results['ego_pose_inv'] = ego_pose_inv
		stacked_imgs = np.stack(results['img'],axis=-1)
		results['img_shape'] = stacked_imgs.shape
		results['ori_shape'] = stacked_imgs.shape
		results['pad_shape'] = stacked_imgs.shape

		omnidrive_answer = get_omnidrive_answer(results, self.model, self.test_pipeline)
		print(f"> OmniDrive: {omnidrive_answer}")

		if PARSER_TYPE == "llama" and SIGNAL_TYPE == "generation":
		   parser_control = llama_chat_generation(self.parser, omnidrive_answer)
		   
		elif PARSER_TYPE == "llama" and SIGNAL_TYPE == "selection":
		   parser_control = llama_chat_selection(self.parser, omnidrive_answer)
		   
		elif PARSER_TYPE == "llava" and SIGNAL_TYPE == "generation":
		   parser_control = llava_chat_generation(self.parser, Image.fromarray(tick_data['imgs']['CAM_FRONT']), omnidrive_answer)

		elif PARSER_TYPE == "llava" and SIGNAL_TYPE == "selection":
		   parser_control = llava_chat_selection(self.parser, Image.fromarray(tick_data['imgs']['CAM_FRONT']), omnidrive_answer)
		   
		else:
		   raise ValueError(f"Unsupported PARSER_TYPE: {PARSER_TYPE}, SIGNAL_TYPE: {SIGNAL_TYPE}")
		
		print(f"> {PARSER_TYPE}_{SIGNAL_TYPE}_control: {parser_control}")

		throttle_traj = parser_control['throttle']
		steer_traj = parser_control['steer']
		brake_traj = parser_control['brake']
		self.pid_metadata = {}
		self.pid_metadata['agent'] = 'only_traj'
		self.pid_metadata['vlm_answer'] = omnidrive_answer
		self.pid_metadata['steer_traj'] = float(steer_traj)
		self.pid_metadata['throttle_traj'] = float(throttle_traj)
		self.pid_metadata['brake_traj'] = float(brake_traj)
		
		if brake_traj < 0.05: brake_traj = 0.0
		if throttle_traj > brake_traj: brake_traj = 0.0
		if tick_data['speed']>5:
		  	throttle_traj = 0
		control = carla.VehicleControl()
		control.steer = np.clip(float(steer_traj), -1, 1)
		control.throttle = np.clip(float(throttle_traj), 0, 0.75)
		control.brake = np.clip(float(brake_traj), 0, 1)
		
		self.pid_metadata['speed'] = tick_data['speed']
		self.pid_metadata['steer'] = control.steer
		self.pid_metadata['throttle'] = control.throttle
		self.pid_metadata['brake'] = control.brake
		metric_info = self.get_metric_info()
		self.metric_info[self.step] = metric_info

		if SAVE_PATH is not None and self.step % 1 == 0:
		  	self.save(tick_data, timestamp)
		return control


	def save(self, tick_data, timestamp):
		frame = self.step // 5
		Image.fromarray(tick_data['imgs']['CAM_FRONT']).save(self.save_path / 'rgb_front' / ('%04d.png' % frame))
		outfile = open(self.save_path / 'meta' / ('%04d.json' % frame), 'w')
		json.dump(self.pid_metadata, outfile, indent=4)
		outfile.close()
		outfile = open(self.save_path / 'metric_info.json', 'w')
		json.dump(self.metric_info, outfile, indent=4)
		outfile.close()


	def destroy(self):
		del self.model
		torch.cuda.empty_cache()


	def gps_to_location(self, gps):
		EARTH_RADIUS_EQUA = 6378137.0
		lat, lon = gps
		scale = math.cos(self.lat_ref * math.pi / 180.0)
		my = math.log(math.tan((lat+90) * math.pi / 360.0)) * (EARTH_RADIUS_EQUA * scale)
		mx = (lon * (math.pi * EARTH_RADIUS_EQUA * scale)) / 180.0
		y = scale * EARTH_RADIUS_EQUA * math.log(math.tan((90.0 + self.lat_ref) * math.pi / 360.0)) - my
		x = mx - scale * self.lon_ref * math.pi * EARTH_RADIUS_EQUA / 180.0
		return np.array([x, y])
