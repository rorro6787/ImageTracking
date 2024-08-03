# ImageTracking

<div align="center">
  <p>
    <a href="https://github.com/ultralytics/assets/releases/tag/v8.2.0" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Vision banner"></a>
  </p>
</div>

This repository contains code and resources for training a computer vision model using the YOLO (You Only Look Once) architecture to detect and track human hands forming different shapes for the rock-paper-scissors game. The project also includes functionality to determine the result of the game based on the detected hand shapes and perform logis associated.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)

## Introduction

The goal of this project is to create a system that can automatically detect and track hands of human beings placed in different shapes for the rock paper scissors game and inference the final result of the game when the input image for our model is composed of 2 hands facing each other in a game. This can be useful for educational purposes, automated grading systems, and more.

## Features

- Detection and tracking of hands using YOLO
- Preprocessing and augmentation of training data
- Training scripts for custom YOLO models
- Evaluation scripts to assess model performance
- Inference the result of the game
- PDF explanation of all the code: [Donwload the PDF document](https://github.com/rorro6787/ImageTracking/blob/gamma/Image_Tracking_YOLO.pdf)

## Requirements

- Python 3.x

## Installation

1. Clone the repository:
   
    ```sh
    git clone https://github.com/yourusername/repository_name.git
    ```
3. Navigate to the project directory:
   
    ```sh
    cd repository_name
    ```
6. (Optional) Create a virtual environment:

   ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On macOS/Linux use 'python -m venv venv
                                                   source venv/bin/activate'
    ```

5. Select venv as your python interpreter (in VSC):
   
    ```sh
    > Python: Select Interpreter
    .\venv\Scripts\python.exe # On macOS/Linux use './venv/bin/python'
    ```
8. Install the required packages:
   
    ```sh
    pip install -r requirements.txt
    ```

7. If you want to do a pull request which implies adding more dependencias, remember to update the requirements file using:
   
    ```sh
    pip freeze > requirements.txt
    ```

## Usage
If you want to use the script that prepares the dataset and trains the model, follow these instructions within the train_model folder, where you will find train_script.py:

1. To prepare the dataset, run the script with the following arguments:
   
    ```sh
    python .\train_script.py 'c'
    ```
2. To remove the entire dataset structure, run the script with the following arguments:

   ```sh
    python .\train_script.py 'r'
    ```
2. To train the model (after the dataset structure has been created), run the script with the following arguments:

   ```sh
    python .\train_script.py 't' --e=number_of_epochs
    ```
If you want to test with the already trained model, navigate to the test_model folder, where you will find test_script.py and use it as follows:

1. To perform inferences on an image containing rock, paper, or scissors, run the script with the following arguments:
   ```sh
    python .\test_script.py 'ii' --source=image_path
    ```
2. To perform inferences on a video containing rock, paper, or scissors, run the script with the following arguments:
   ```sh
    python .\test_script.py 'iv' --source=video_path
    ```



## Dataset

The dataset should consist of video frames or images containing hands of human persons forming the different forms, annotated with bounding boxes. You can use existing datasets like the ones on Roboflow web or annotate you own dataset by hand. If you successfully train the model with your dataset, it can achieve accuracy comparable to ours in tracking rock, paper, scissors.

## Contributors
- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/rorro6787) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/emilio-rodrigo-carreira-villalta-2a62aa250/) **Emilio Rodrigo Carreira Villalta**
- [![GitHub](https://img.shields.io/badge/GitHub-100000?style=flat&logo=github&logoColor=white)](https://github.com/javimp2003uma) [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/javier-montes-p%C3%A9rez-a9765a279/) **Javier Montes Pérez**

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## Acknowledgements

- Inspired by various tutorials and resources on the YOLO documentation



