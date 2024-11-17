# EE260C_LAB1: Perception

## Objective
This lab implements the perception module with a focus on 3D detection. The goal is to detect 3D bounding boxes around traffic participants using single or multiple modality sensors. The solution will be evaluated against ground truth in CARLA.

### Getting Started
To get started, I accepted the assignment via Github:  https://classroom.github.com/a/w3FoW0fO 

### Part 1: Environment setup
This lab is based on a modified version of automatic_control.py, which now relies on the output of the detector.py instead of correctly detecting objects and obstacles. 

I Started the environment by 
Running carla server
Generating background traffic
vglrun python generate_traffic.py using BCOE system
Running the automatic control agent
vglrun python automatic_control.py 

By the end of the route, or whenever you terminate it, the terminal will print the total number of object instances to be detected, and the average precision of different IoUs.

![Screenshot 2024-10-27 at 1 10 47 AM](https://github.com/user-attachments/assets/5e3b260e-8f98-4353-b99a-e2fa2d4f6185)



### Part 2: Setup sensor and visualize groundtruth

Used the following sensor setup:
sensors = [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 1280, 'height': 720, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
                      'range': 50, 
                      'rotation_frequency': 20, 'channels': 64,
                      'upper_fov': 4, 'lower_fov': -20, 'points_per_second': 2304000,
                      'id': 'LIDAR'},

            {'type': 'sensor.other.gnss', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'id': 'GPS'}
        ]
        
### Visualizing various sensors

![Screenshot 2024-11-02 at 12 47 33 AM (2)](https://github.com/user-attachments/assets/e24a4323-65af-4a0d-807b-84c0c69f4d04)

![Screenshot 2024-11-02 at 12 48 10 AM](https://github.com/user-attachments/assets/914a9760-ee11-4eb5-82b5-ed26ca245fb2)

![Screenshot 2024-11-02 at 12 48 33 AM](https://github.com/user-attachments/assets/a06a981a-341b-414e-9afa-2428c5b6a630)


![Screenshot 2024-11-02 at 12 48 53 AM](https://github.com/user-attachments/assets/9acc1045-766a-49c0-85be-0d0c4290e687)

### Part 3: Integrate a pretrained perception model

Detect function takes an input argument “sensor_data” and output detection results. This part fills in this function with a pre-trained perception model.
I tried MM3D unsuccessfully due to problems with the GPU machine.
Therefore, I used FasterRCNN pretrained object detector from PyTorch and added it to carla.
I manipulated the sensor data to be suitable for the FasterRCNN model.

![Screenshot 2024-11-17 at 3 28 00 PM](https://github.com/user-attachments/assets/a06a2943-0b4d-4a90-9db7-f9f9eb47ad6b)

![Screenshot 2024-11-17 at 3 28 20 PM](https://github.com/user-attachments/assets/f438c5df-4411-4213-9d91-e928e7eb668f)

![Screenshot 2024-11-17 at 3 28 33 PM](https://github.com/user-attachments/assets/53ef2d7a-010a-40ee-8f21-18f3a9018db5)



