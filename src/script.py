import cv2
import os
import sys
from ultralytics import YOLO
import shutil
import random
from zipfile import ZipFile
import gdown

cd = os.getcwd()                    # it's gonna be /src
base_path = 'dataset_split'

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

def remove_empty_dirs(path):
    """
    Removes empty directories including the base directory if it's empty.

    Args:
        path (str): Path to the base directory.
    """
    for dirpath, dirnames, filenames in os.walk(path, topdown=False):
        if not dirnames and not filenames:
            os.rmdir(dirpath)

    # Attempt to remove the base directory
    try:
        os.rmdir(path)
    except OSError as e:
        print()

def removeAll():
    if os.path.exists(f"{cd}/dataset_split"):
        shutil.rmtree(f'{cd}/dataset_split', ignore_errors=True)
        print("Data distribution deleted successfully ...")
    else:
        print(f"The directory {cd}/dataset_split does not exist.")
    if os.path.exists(f"{cd}/dataset.zip"):
        os.remove(f"{cd}/dataset.zip")
        print("dataset.zip deleted successfully ...")
    else:
        print(f"The file {cd}/dataset_zip does not exist.")
    if os.path.exists(f"{cd}/dataset"):
        shutil.rmtree(f'{cd}/dataset', ignore_errors=True)
        print("Data distribution deleted successfully ...")
    else:
        print(f"The directory {cd}/dataset does not exist.")

def prepare_structure():
    """
    Prepares the directory structure for the rock_paper_scissors dataset and splits the data into training, validation, and test sets.

    Unzips the dataset, shuffles the data, splits it into training (60%), validation (30%), and test (10%) sets, 
    and moves the files into the corresponding directories.
    """

    # Define paths
    zip_file_path = 'dataset.zip'
    base_extract_path = 'dataset'
    url = 'https://drive.google.com/uc?id=1h7PrvmW8SaI6wyGE1x2bD-nMirccyfcr'
    gdown.download(url,zip_file_path,quiet=False)
    
    # Unzip the file
    with ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(base_extract_path)

    # Paths for extracted images and labels
    images_path = os.path.join(base_extract_path, 'all_images')
    labels_path = os.path.join(base_extract_path, 'all_labels')

    # Gather image and label file names and sort them in ascending order (in this case string order)
    image_files = sorted(os.listdir(images_path))
    label_files = sorted(os.listdir(labels_path))
    assert len(image_files) == len(label_files) == 6759 # (alredy known number of items in dataset)

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

    # Remove empty directories
    remove_empty_dirs(base_extract_path)

    # Create config.yaml file
    config_content = f"""path: {os.path.join(cd, 'dataset_split')}\ntrain: train\nval: val\ntest: test\n\nnc: 3\nnames: ['paper', 'rock', 'scissors']"""

    with open(os.path.join(base_path, 'config.yaml'), 'w') as file:
        file.write(config_content)

    print("Data distribution and yaml creation completed successfully ...")


def train_model(yaml_file, epochs, project):
    """
    Trains the YOLO model on the provided dataset and saves the training metrics.

    Args:
        yaml_file (str): Path to the YOLO configuration YAML file.
        epochs (int): Number of training epochs.
        project (str): Path to the project directory where the training results will be saved.
    """

    model = YOLO(model="yolov8n.pt", task="detect")
    model.to('cuda')
    model.train(data = yaml_file,
        	  epochs = epochs,
              project = project,
        	  batch = 8,
              name = "train")          # batch = 8 is neccesary because GPU does not support higher
    
    results2 = model.val(data = yaml_file,
                        project = project,
                        batch = 8,
                        name = "test",
                        split = "test")
    
    # Create full route for the metrics file
    metrics_file_path2 = os.path.join(project, "testing_metrics.txt")
        
    # Write metrics in that file
    with open(metrics_file_path2, 'w') as f:
        f.write("Another metrics:\n")
        f.write(" Average precision for all classes: {}\n".format(results2.box.all_ap))
        f.write(" Average precision: {}\n".format(results2.box.ap))
        f.write(" Average precision at IoU=0.50: {}\n".format(results2.box.ap50))
        f.write(" F1 score: {}\n".format(results2.box.f1))
        f.write(" Mean average precision: {}\n".format(results2.box.map))
        f.write(" Mean average precision at IoU=0.50: {}\n".format(results2.box.map50))
        f.write(" Mean average precision at IoU=0.75: {}\n".format(results2.box.map75))
        f.write(" Mean average precision for different IoU thresholds: {}\n".format(results2.box.maps))
        f.write(" Mean precision: {}\n".format(results2.box.mp))
        f.write(" Mean recall: {}\n".format(results2.box.mr))
        f.write(" Precision: {}\n".format(results2.box.p))
        f.write(" Precision values: {}\n".format(results2.box.prec_values))
        f.write(" Recall: {}\n".format(results2.box.r))


if len(sys.argv) == 2 and 'c' in sys.argv:
    prepare_structure()
elif len(sys.argv) == 3 and 't' in sys.argv:
    if sys.argv[-1].startswith("--e="):
        # Remove "--e=" prefix
        nEp = int(sys.argv[-1][4:])
        train_model(f"{cd}/dataset_split/config.yaml",nEp,"Hands tracking")
    else:
        # Handle invalid argument format
        print(f"Invalid argument format: {sys.argv[-1]} or {sys.argv[-3]} :( Expected format: --e=nEpochs...")
elif len(sys.argv) == 2 and 'r' in sys.argv:
    removeAll()