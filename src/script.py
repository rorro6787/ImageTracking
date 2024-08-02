import cv2
import os
import sys
from ultralytics import YOLO
import shutil
import random
from zipfile import ZipFile
import gdown
import time
import emoji

cd = os.getcwd()                    # it's gonna be /src
base_path = 'dataset_split'
classes = ['paper', 'rock', 'scissors']

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
    """
    Removes the dataset_split directory, dataset.zip file, and dataset directory if they exist.
    Provides feedback if these items are not found.
    """

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

def inference_over_image(sourcePath):
    """
    Performs inference on a single image using a pre-trained YOLO model and prints the result.

    Args:
        sourcePath (str): Path to the image file to be analyzed.
    """

    file_path = 'best.pt'
    urlModelDrive = 'https://drive.google.com/uc?id=13v14-RVorjtMCB5vx0qktgPspk2-t_jT'
    if not os.path.exists(file_path):
        gdown.download(urlModelDrive,file_path,quiet=False)
    else:
        print("best.pt alredy exists ...")

    model = YOLO(file_path)
    results = model(sourcePath, save = True)

    if len(results[0].boxes) != 2:
        print(":( In your image there are not 2 hands!")
    else:
        print(":) You have input a correct image!")
        print(f"2 hands found with confidences: {results[0].boxes[0].conf.item()} and {results[0].boxes[1].conf.item()}\n")
        print(f"{results[0].boxes[0].xyxy.tolist()[0][1]} and {results[0].boxes[1].xyxy.tolist()[0][1]}")
        if results[0].boxes[0].xyxy.tolist()[0][1] < results[0].boxes[1].xyxy.tolist()[0][1]:
            hand1 = results[0].boxes[0].cls
            hand2 = results[0].boxes[1].cls
        else:
            hand1 = results[0].boxes[1].cls
            hand2 = results[0].boxes[0].cls  
        determine_winner(hand1,hand2)
        results[0].show()


def determine_winner(hand1, hand2):
    """
    Determines the winner of a rock-paper-scissors game based on the given hands and prints the result.

    Args:
        hand1 (str): The class of the first hand (e.g., 'Rock', 'Paper', 'Scissors').
        hand2 (str): The class of the second hand (e.g., 'Rock', 'Paper', 'Scissors').
    """
    hand1 = classes[int(hand1.item())]
    hand2 = classes[int(hand2.item())]

    print(f"Facing {hand1} vs {hand2}\U0001f600")
    time.sleep(1)

    if hand1 == hand2:
        print("It's a draw!")
        print(f"Both hands are {hand1}.")
    elif (hand1 == "rock" and hand2 == "scissors") or \
            (hand1 == "scissors" and hand2 == "paper") or \
            (hand1 == "paper" and hand2 == "rock"):
        print("WE HAVE A WINNER!")
        print(f"{hand1} beats {hand2} :)".upper())
        print(emoji.emojize(":fire:"))
    else:
        print("WE HAVE A WINNER!")
        print(f"{hand2} beats {hand1} :)".upper())
        print(emoji.emojize(":fire:"))


def consistency(pairs):
    """
    Checks if all pairs in the list are consistent, meaning they match the first pair in either order.

    Args:
        pairs (list): List of tuples where each tuple represents a pair of classes.

    Returns:
        bool: True if all pairs are consistent, False otherwise.
    """

    first_pair = pairs[0]
    for pair in pairs[1:]:
        if not ((pair[0] == first_pair[0] and pair[1] == first_pair[1]) or
                (pair[0] == first_pair[1] and pair[1] == first_pair[0])):
            return False
    return True


def inference_over_video(sourcePath, frame_buffer_size=5):
    """
    Performs inference on a video file, analyzing pairs of detected hands across frames and determining the winner if pairs are consistent.

    Args:
        sourcePath (str): Path to the video file to be analyzed.
        frame_buffer_size (int): Number of frames to buffer for consistency checking. Default is 5.
    """

    file_path = 'best.pt'
    urlModelDrive = 'https://drive.google.com/uc?id=1h7PrvmW8SaI6wyGE1x2bD-nMirccyfcr'  # Cambia la URL si es necesario
    if not os.path.exists(file_path):
        gdown.download(urlModelDrive,file_path,quiet=False)
    else:
        print("best.pt alredy exists ...")

    cap = cv2.VideoCapture(sourcePath)
    model = YOLO(file_path)

    # Buffer para almacenar pares de clases detectadas
    frame_buffer = []

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            # Ejecuta el seguimiento con YOLOv8 en el frame, persistiendo las pistas entre frames
            results = model.track(frame, persist=True)

            if results[0].boxes:
                # Asegúrate de que haya exactamente 2 manos detectadas
                if len(results[0].boxes) == 2:
                    hand1_cls = results[0].boxes[0].cls
                    hand2_cls = results[0].boxes[1].cls
                    frame_buffer.append((hand1_cls, hand2_cls))

                    if len(frame_buffer) > frame_buffer_size:
                        frame_buffer.pop(0)

                    # Verifica la consistencia después de llenar el buffer
                    if len(frame_buffer) == frame_buffer_size:
                        if consistency(frame_buffer):
                            # Determina el ganador basado en el par consistente
                            determine_winner(frame_buffer[0][0], frame_buffer[0][1])
                            frame_buffer = []  # Reinicia el buffer después de determinar el ganador
                            break
                        else:
                            print("Las detecciones no son consistentes, esperando más frames...")
                else:
                    # Si no hay exactamente 2 manos detectadas, reinicia el buffer
                    frame_buffer = []

            # Visualiza los resultados en el frame
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Rompe el bucle si se presiona 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Rompe el bucle si se llega al final del video
            break

    # Libera el objeto de captura de video y cierra la ventana de visualización
    cap.release()
    cv2.destroyAllWindows()


if len(sys.argv) == 2 and 'c' in sys.argv:
    prepare_structure()
elif len(sys.argv) == 3 and 't' in sys.argv:
    if sys.argv[-1].startswith("--e="):
        # Remove "--e=" prefix
        nEp = int(sys.argv[-1][4:])
        train_model(f"{cd}/dataset_split/config.yaml",nEp,"Hands tracking")
    else:
        # Handle invalid argument format
        print(f"Invalid argument format: {sys.argv[-1]}  :( Expected format: --e=nEpochs...")
elif len(sys.argv) == 2 and 'r' in sys.argv:
    removeAll()
elif len(sys.argv) == 3 and 'ii' in sys.argv:
    if sys.argv[-1].startswith("--source="):
        # Remove "--source=" prefix
        sourcePath = sys.argv[-1][9:]
        inference_over_image(sourcePath)
    else:
        # Handle invalid argument format
        print(f"Invalid argument format: {sys.argv[-1]}  :( Expected format: --source=sourcePath...")
elif len(sys.argv) == 3 and 'iv' in sys.argv:
    if sys.argv[-1].startswith("--source="):
        # Remove "--source=" prefix
        sourcePath = sys.argv[-1][9:]
        inference_over_video(sourcePath)
    else:
        # Handle invalid argument format
        print(f"Invalid argument format: {sys.argv[-1]}  :( Expected format: --source=sourcePath...")