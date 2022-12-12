# Self-Driving-Car
The objective of the autonomous vehicle is to decide on the steering angle to apply along a given path. Dataset used here is Sully Chen Dataset. A webcam was placed behind the windshield for recording the video. A drive of 25 minutes has been recorded. The video is segmented at the measure of 30 frames per second. The approach is interfacing with the car directly, accessing the CAN-BUS using the OBD-II port that every modern car is equipped with. CAN BUS signals were decoded before. CAN BUS signals tells every part of automobile connected to CANBUS. We can just identify the steering angle from the particular column. More details about the dataset collection can be accessed here. The objective here is to make CNN learn and extract features such as signals, weather conditions, lane marking, path planning just be providing steering angle as train data and we have obtained it to a considerable satisfaction. CNN is used for feature extraction and FC layers were used for predicting steering angle.

INPUTS IN SELF DRIVING CAR:
•	Cameras (Front, Rear, Side views)
•	LIDAR for Lane Detection and Marking
•	Ultrasonic Sensor to measure the position of objects close to vehicle.
•	RADAR to measure distance.
•	GPS for positioning from satellites etc.,
OUTPUTS:
•	Steering wheel angle
•	Acceleration
•	Brakes
•	Indicators
•	Wipers
•	Horn and more
Here, In this project we take input video from front camera and we predict steering wheel angle.




![image](https://user-images.githubusercontent.com/78693179/207123748-220cb3c1-7954-4c5f-a79b-84bc5dc528e9.png)


Also, Refer:
https://towardsdatascience.com/how-a-high-school-junior-made-a-self-driving-car-705fa9b6e860
https://github.com/SullyChen/Autopilot-TensorFlow
https://arxiv.org/pdf/1604.07316.pdf


