import os
import pickle
import mediapipe as mp
import cv2
from collect_data import DATA_DIR

from train_classifier import DATA_PICKLE_PATH


def initialize_hands():
    """
    Initialize MediaPipe Hands module.
    :return: The Hands object initialized.
    """
    return mp.solutions.hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


def process_image(hands, img_path):
    """
    Load and process an image, returning hand landmark coordinates.
    :param hands: MediaPipe Hands object.
    :param img_path: the path of the image.
    :return: The hand landmark coordinates of the MediaPipe Hands object.
    """
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    return extract_landmarks(results) if results.multi_hand_landmarks else None


def extract_landmarks(results):
    """
    Extract (x, y) coordinates of hand landmarks.
    :param results: An image processed to recognize hands using the MediaPipe Hands module.
    :return: The extracted (x,y) coordinates of the hands landmarks.
    """
    x_ = []
    y_ = []
    hands_data = []
    for hand_landmarks in results.multi_hand_landmarks:
        data_aux = []
        for landmark in hand_landmarks.landmark:
            x_.append(landmark.x)
            y_.append(landmark.y)

        width = max(x_) - min(x_)
        height = max(y_) - min(y_)

        for landmark in hand_landmarks.landmark:
            x = (landmark.x - min(x_)) / width
            y = (landmark.y - min(y_)) / height
            data_aux.append(x)
            data_aux.append(y)

        hands_data.append(data_aux)
    return hands_data


def save_dataset(data, labels):
    """
    Save dataset to a pickle file.
    :param data: The data extracted from the images.
    :param labels: The label of the extracted data

    """
    with open(DATA_PICKLE_PATH, 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print(f"Dataset saved to {DATA_PICKLE_PATH}")


def generate_dataset(data_dir=DATA_DIR):
    """
    Generate dataset recognizing different classes of hand gestures using images.
    :param data_dir: a directory containing directories of different classes of hand gestures.
    """
    hands = initialize_hands()
    data, labels = [], []

    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)

        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            landmarks_list = process_image(hands, img_path)

            if landmarks_list:
                for landmarks in landmarks_list:
                    data.append(landmarks)
                    labels.append(label)
            else:
                print(f"No hands detected in {img_name}")

    save_dataset(data, labels)


# generate_dataset()