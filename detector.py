import torch
import torchvision
import numpy as np
import pygame


class Detector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights='DEFAULT')
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode


    def sensors(self):  # pylint: disable=no-self-use
        """
        Define the sensor suite required by the detector. The location is defined with respect to the actor center
        -- x axis is longitudinal (forward-backward)
        -- y axis is lateral (left and right)
        -- z axis is vertical
        Unit is in meters


        :return: a list containing the required sensors in the following format:


        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},


            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},


            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]


        """
        sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},


            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},


            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                      'range': 50, 'rotation_frequency': 20, 'channels': 64, 'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,'id': 'LIDAR'}
        ]


        return sensors


    def detect(self, sensor_data):
        camera_data = sensor_data.get('Left')
        if camera_data is None:
            return {}
        frame_id, image = camera_data 

        image = image[:, :, :3]

        image = torch.tensor(image).permute(2, 0, 1).float().to(self.device)  
        image = image / 255.0 
        input_tensor = image.unsqueeze(0)  


        with torch.no_grad():
            results = self.model(input_tensor)

        det_boxes = []
        det_class = []
        det_score = []

        for box, label, score in zip(results[0]['boxes'], results[0]['labels'], results[0]['scores']):
            if score > 0.2:  # Confidence threshold
                box_np = box.cpu().numpy()
                box_3d = self.convert_to_3d(box_np)
                print("3D bbox:", box_3d)
                det_boxes.append(box_3d)
                det_class.append(int(label.cpu().numpy()) - 1) 
                det_score.append(score.cpu().numpy())


        # Convert lists to numpy arrays in the required format
        return {
            "det_boxes": np.array(det_boxes),
            "det_class": np.array(det_class),
            "det_score": np.array(det_score),
        }




    def convert_to_3d(self, box):
       
        x1, y1, x2, y2 = box
        z = 20.0  # Default depth, adjust as needed


        box_3d = np.array([
            [x1, y1, 0], [x2, y1, 0], [x2, y2, 0], [x1, y2, 0],  
            [x1, y1, z], [x2, y1, z], [x2, y2, z], [x1, y2, z]   
        ])
        return box_3d
