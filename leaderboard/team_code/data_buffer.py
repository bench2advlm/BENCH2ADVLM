from PIL import Image
from collections import deque

class DataBuffer:
    def __init__(self, buffer_size, img_keys=None, sensor_keys=None):
        """
        Initialize a circular buffer for storing historical frames.
        
        Args:
            buffer_size: Maximum number of historical frames to store
            img_keys: List of camera keys to preserve image data, e.g., ['CAM_FRONT', 'CAM_BACK']
            sensor_keys: List of non-image sensor signals to preserve, e.g., ['speed', 'gps']
        """
        self.buffer = deque(maxlen=buffer_size)
        self.img_keys = img_keys if img_keys is not None else []
        self.sensor_keys = sensor_keys if sensor_keys is not None else []


    def add(self, tick_data):
        """
        Filter and store required fields from tick_data into the buffer queue.
        
        Args:
            tick_data: Dictionary from tick() function containing sensor data
        """
        frame_data = {}
        if self.img_keys:
            frame_data['imgs'] = {}
            for key in self.img_keys:
                if key in tick_data.get('imgs', {}):
                    pil_img = Image.fromarray(tick_data['imgs'][key])
                    frame_data['imgs'][key] = pil_img
        for key in self.sensor_keys:
            if key in tick_data:
                frame_data[key] = tick_data[key]
        self.buffer.append(frame_data)


    def get_input(self):
        """
        Retrieve all buffered frame data in chronological order (oldest first).
        
        Returns:
            List of frame data dictionaries
        """
        return list(self.buffer)
        