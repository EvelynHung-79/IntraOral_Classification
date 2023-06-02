# IntraOral_classification
This is an image classification project on intraoral images.

## Dataset
The dataset used is 'angle' with classes of 'frontal', 'frontal_90', 'frontal_180', 'frontal_270', 'left', 'left_90', 'left_180', 'left_270', 'lower_v2', 'lower_90_v2', 'lower_180_v2', 'lower_270_v2', 'others', 'right', 'right_90', 'right_180', 'right_270', 'upper_v2', 'upper_90_v2', 'upper_180_v2', and 'upper_270_v2'.
Data is provided below, and anyone with a dentall account should be able to access.

[https://drive.google.com/drive/folders/13QqLrhO5vA3QhRs47QQk38rzg-kWn0a2?usp=sharing](https://drive.google.com/drive/folders/13QqLrhO5vA3QhRs47QQk38rzg-kWn0a2?usp=sharing)

## Preparation
Prepare the tensorflow gpu environment on your device, and make sure it is running correctly.
The whole training processes for 40 epochs might take more than 8 hours, depending on your GPU strength.
Download python packages if necessary.
And clone this project into your directory to run.

## Model 
The model used is EfficientNetB4 with some specific hyper-parameters, please refer to the python file for more details.

## Run
To re-generate the result, run 'intraoral_train.py' in 'code' folder directly, modify the path if necessary.

## Result
After you have execute the code, 'accuracy graph', 'loss graph', 'training log', 'model architecture', 'model weights', and 'checkpoints' will be saved.
The result is currently 0.99 on F1-score/Precision/Recall/Accuracy (The graph below shows the result of accuracy over best epochs).
And the model is saved in the link below.

https://drive.google.com/drive/folders/13QqLrhO5vA3QhRs47QQk38rzg-kWn0a2?usp=sharing

<img width="366" alt="image" src="https://user-images.githubusercontent.com/57160523/191674959-f27c4795-53d8-4afa-a572-11a517e3a867.png">


