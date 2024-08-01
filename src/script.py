import cv2
import os
import sys
from ultralytics import YOLO
import shutil
import random
from zipfile import ZipFile

cd = os.getcwd()
base_path = 'MNISTyolov8_split'

def move_files(data, split, images_path, labels_path):
    """
    Moves files to their respective directories based on the split.

    Args:
        data (list): List of tuples containing image and label filenames.
        split (str): The type of split ('train', 'val', 'test').
        images_path (str): Path to the directory containing images.
        labels_path (str): Path to the directory containing labels.
    """

    for img_file, lbl_file in data:
        shutil.move(os.path.join(images_path, img_file), os.path.join(base_path, split, 'images', img_file))
        shutil.move(os.path.join(labels_path, lbl_file), os.path.join(base_path, split, 'labels', lbl_file))

def prepare_structure():
    """
    Prepares the directory structure for the MNIST dataset and splits the data into training, validation, and test sets.

    Unzips the dataset, shuffles the data, splits it into training (60%), validation (30%), and test (10%) sets, 
    and moves the files into the corresponding directories.
    """

    # Define paths
    zip_file_path = 'MNISTyolov8.zip'
    base_extract_path = 'MNISTyolov8'
    
    # Unzip the file
    with ZipFile(f"MNIST.v5i.yolov8/{zip_file_path}", 'r') as zip_ref:
        zip_ref.extractall(base_extract_path)

    # Paths for extracted images and labels
    images_path = os.path.join(base_extract_path, 'all_images')
    labels_path = os.path.join(base_extract_path, 'all_labels')

    # Gather image and label file names and sort them in ascending order (in this case string order)
    image_files = sorted(os.listdir(images_path))
    label_files = sorted(os.listdir(labels_path))
    assert len(image_files) == len(label_files) == 7128 # (alredy known number of items in dataset)

    # Shuffle and split data. Data is a list of tuples ...
    data = list(zip(image_files, label_files))
    random.shuffle(data)

    train_data = data[:int(0.6 * len(data))]
    val_data = data[int(0.6 * len(data)):int(0.9 * len(data))]
    test_data = data[int(0.9 * len(data)):]

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_path, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(base_path, split, 'labels'), exist_ok=True)
    
    move_files(train_data,'train',images_path,labels_path)
    move_files(val_data,'val',images_path,labels_path)
    move_files(test_data,'test',images_path,labels_path)

    print("Data distribution completed successfully ...")


def train_model(yaml_file, epochs, project):
    """
    Trains the YOLO model on the provided dataset and saves the training metrics.

    Args:
        yaml_file (str): Path to the YOLO configuration YAML file.
        epochs (int): Number of training epochs.
        project (str): Path to the project directory where the training results will be saved.
    """

    model = YOLO(model="yolov8x.pt", task="detect")
    model.to('cuda')
    results = model.train(data = yaml_file,
        	  epochs = epochs,
              project = project,
        	  batch = 8,
              name = "train")          # batch = 8 is neccesary because GPU does not support higher
    
    # Crear la ruta completa para el archivo de texto
    metrics_file_path = os.path.join(project, "training_metrics.txt")
        
    # Escribir las métricas en el archivo de texto
    with open(metrics_file_path, 'w') as f:
        f.write("Another metrics:\n")
        f.write(" Average precision for all classes: {}\n".format(results.box.all_ap))
        f.write(" Average precision: {}\n".format(results.box.ap))
        f.write(" Average precision at IoU=0.50: {}\n".format(results.box.ap50))
        f.write(" F1 score: {}\n".format(results.box.f1))
        f.write(" Mean average precision: {}\n".format(results.box.map))
        f.write(" Mean average precision at IoU=0.50: {}\n".format(results.box.map50))
        f.write(" Mean average precision at IoU=0.75: {}\n".format(results.box.map75))
        f.write(" Mean average precision for different IoU thresholds: {}\n".format(results.box.maps))
        f.write(" Mean precision: {}\n".format(results.box.mp))
        f.write(" Mean recall: {}\n".format(results.box.mr))
        f.write(" Precision: {}\n".format(results.box.p))
        f.write(" Precision values: {}\n".format(results.box.prec_values))
        f.write(" Recall: {}\n".format(results.box.r))



if len(sys.argv) == 2 and 'c' in sys.argv:
    prepare_structure()
elif len(sys.argv) == 2 and 't' in sys.argv:
    train_model(f"{cd}/config.yaml",100,"Digits tracking")