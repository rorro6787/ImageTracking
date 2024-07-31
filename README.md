# ImageTracking

<div align="center">
  <p>
    <a href="https://github.com/ultralytics/assets/releases/tag/v8.2.0" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="YOLO Vision banner"></a>
  </p>
</div>

This repository contains code and resources for training a computer vision model using the YOLO (You Only Look Once) architecture to detect and track handwritten digits in videos. The project also includes functionality to perform arithmetic operations based on the detected digits.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Contributing](#contributing)

## Introduction

The goal of this project is to create a system that can automatically detect and track handwritten digits in video frames and calculate operations based on those digits. This can be useful for educational purposes, automated grading systems, and more.

## Features

- Detection and tracking of handwritten digits in video using YOLO
- Preprocessing and augmentation of training data
- Training scripts for custom YOLO models
- Evaluation scripts to assess model performance
- Arithmetic operations on detected digits
- PDF explanation of all the code: [Donwload the PDF document]()

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
5. (Optional) Create a virtual environment:
   
    ```sh
    python -m venv venv
    .\venv\Scripts\activate  # On macOS/Linux use 'python -m venv venv
                                                   source venv/bin/activate'
    ```
7. Install the required packages:
   
    ```sh
    pip install -r requirements.txt
    ```

7. If you want to do a pull request which implies adding more dependencias, remember to update the requirements file using:
   
    ```sh
    pip freeze > requirements.txt
    ```

## Usage

## Dataset

The dataset should consist of video frames or images with handwritten digits, annotated with bounding boxes. You can use existing datasets like MNIST and augment them to create video sequences.

## Contributors
- [Emilio Rodrigo Carreira Villalta](https://github.com/rorro6787)
- [Javier Montes Pérez](javimp2003uma)

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature-branch`)
5. Create a new Pull Request

## Acknowledgements

- Inspired by various tutorials and resources on the YOLO documentation



