import cv2
import os
import numpy as np
import json
from tqdm import trange
import argparse
import textwrap

def parse_args():
    parser = argparse.ArgumentParser(description="Generate video from images and metadata")
    parser.add_argument('-f', '--images_folder', type=str, required=True, help="Path to the folder containing images and metadata")
    parser.add_argument('-o', '--output_video', type=str, default='./output_video.mp4', help="Output video file path")
    parser.add_argument('--fps', type=int, default=10, help="Frames per second")
    parser.add_argument('--font_scale', type=int, default=1, help="Font scale for text")
    parser.add_argument('--text_color', type=tuple, default=(255, 255, 255), help="Text color in RGB format")
    parser.add_argument('--text_position', type=tuple, default=(50, 50), help="Text position in the image")
    return parser.parse_args()


def create_video(images_folder, output_video, fps, font_scale, text_color, text_position):
    images = [img for img in os.listdir(os.path.join(images_folder, 'rgb_front')) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort()

    frame = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), images[0]))
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for i in trange(1, len(images)):
        image = images[i]
        f = open(os.path.join(images_folder, f'meta/{i:04}.json'), 'r')
        meta = json.load(f)
        steer = float(meta['steer'])
        throttle = float(meta['throttle'])
        brake = float(meta['brake'])
        vlm_answer = meta['vlm_answer']
        speed = float(meta['speed'])
        img = cv2.imread(os.path.join(os.path.join(images_folder, 'rgb_front'), image))

        vlm_wrapped = textwrap.wrap(f"ADVLM: {vlm_answer}", width=90)
        vlm_x, vlm_y = 50, 50
        line_height = 30
        for line in vlm_wrapped:
            cv2.putText(img, line, (vlm_x, vlm_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)
            vlm_y += line_height
        
        control_text = f'speed: {round(speed,2)}, steer: {round(steer,2)}, throttle: {round(throttle,2)}, brake: {round(brake,2)}'
        control_y = img.shape[0] - 50
        cv2.putText(img, control_text, (50, control_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 2, cv2.LINE_AA)

        video.write(img)
    video.release()

if __name__ == "__main__":
    args = parse_args()
    
    create_video(
        args.images_folder,
        args.output_video,
        args.fps,
        args.font_scale,
        args.text_color,
        args.text_position
    )