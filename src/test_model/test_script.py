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

    results = model(sourcePath, save = True, project=os.path.join(cd, 'predictions', 'images'))

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


def inference_over_video(sourcePath, outputPath=os.path.join(cd, 'predictions', 'videos', str('output' + str(random.randint(0, 100)))) + '.mp4', frame_buffer_size=5):
    file_path = 'best.pt'
    urlModelDrive = 'https://drive.google.com/uc?id=1h7PrvmW8SaI6wyGE1x2bD-nMirccyfcr'
    
    if not os.path.exists(file_path):
        gdown.download(urlModelDrive, file_path, quiet=False)
    else:
        print("best.pt already exists ...")

    cap = cv2.VideoCapture(sourcePath)
    model = YOLO(file_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Intenta usar 'XVID' para .avi o 'mp4v' para .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

    # Verifica si el VideoWriter se ha inicializado correctamente
    if not out.isOpened():
        print(f"Error: No se pudo abrir el archivo de salida: {outputPath}")
        cap.release()
        return

    frame_buffer = []

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            results = model.track(frame, persist=True)

            if results[0].boxes:
                if len(results[0].boxes) == 2:
                    hand1_cls = results[0].boxes[0].cls
                    hand2_cls = results[0].boxes[1].cls
                    frame_buffer.append((hand1_cls, hand2_cls))

                    if len(frame_buffer) > frame_buffer_size:
                        frame_buffer.pop(0)

                    if len(frame_buffer) == frame_buffer_size:
                        if consistency(frame_buffer):
                            determine_winner(frame_buffer[0][0], frame_buffer[0][1])
                            frame_buffer = []
                            break
                        else:
                            print("Las detecciones no son consistentes, esperando más frames...")
                else:
                    frame_buffer = []

            annotated_frame = results[0].plot()

            out.write(annotated_frame)
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    if len(sys.argv) == 3 and 'ii' in sys.argv:
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

if __name__ == "__main__":
    main()