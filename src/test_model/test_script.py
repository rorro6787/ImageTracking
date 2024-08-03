import cv2
import os
import sys
from ultralytics import YOLO
import shutil
import random
from zipfile import ZipFile
import gdown
import time

cd = os.getcwd()
classes = ['paper', 'rock', 'scissors']
model = YOLO(os.path.join(cd, 'best.pt'))

def inference_over_image(sourcePath):
    """
    Performs inference on a single image using a pre-trained YOLO model and prints the result.

    Args:
        sourcePath (str): Path to the image file to be analyzed.
    """

    results = model(sourcePath, save = True, project=cd)

    if len(results[0].boxes) != 2:
        print(":( In your image there are not 2 hands!")
    else:
        print(":) You have input a correct image!")
        print(f"2 hands found with confidences: {results[0].boxes[0].conf.item()} and {results[0].boxes[1].conf.item()}\n")
        hand1 = results[0].boxes[0].cls
        hand2 = results[0].boxes[1].cls
        determine_winner(hand1,hand2)

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
        print("Hand 1 wins!")
        print(f"{hand1} beats {hand2} :)")
    else:
        print("Hand 2 wins!")
        print(f"{hand2} beats {hand1} :)")


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

#print(os.path.join(cd, 'tests', '1590047761696.jpg'))
inference_over_image(os.path.join(cd, 'tests', '1590047761696.jpg'))

