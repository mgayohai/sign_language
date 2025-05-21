import os
import numpy as np
import pickle
import cv2
import mediapipe as mp
from train_classifier import MODEL_DIR, create_model
from collect_data import collect_data, display_text
from create_dataset import generate_dataset


def get_language_model(language_dir):
    """
    Loads a sign language prediction model given a directory.
    :param language_dir:The sign language directory.
    :return: The loaded model.
    """
    model_dict = pickle.load(open(language_dir, 'rb'))
    model = model_dict['model']
    return model


def generate_hands():
    """
    Generates hand artifacts the program uses to transfer real-time feed to sign language.
    :return: the hand artifacts.
    """
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    return mp_hands, mp_drawing, mp_drawing_styles


def get_sign_language(cap):
    """
    Gets the sign language classification (as in ASL. ) we want to translate in real time.
    :param ret: The video capture' ret object.
    :param frame: The video capture's frame object.
    :return: The sign language name.
    """
    language = ''

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display_text(frame, f"Please enter requested sign language classification: {language}")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key
            return None
        elif key == 13:  # Enter key
            if language == '':
                continue
            return language
        elif key == 8:  # Backspace key
            language = language[:-1]
        elif key != 255:
            language += chr(key).upper()


def translate_in_real_time(cap, model, language):
    """
    Given a prediction model of a sign language uses it to translate real-time feed to the corresponding letters.
    :param cap: The capture artifact of the video feed.
    :param model: The prediction model of the given language.
    :param language: The given language we want to translate.
    :return:
    """
    print(f'Starting translation of {language}')
    mp_hands, mp_drawing, mp_drawing_styles = generate_hands()
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    while True:
        ret, frame = cap.read()
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key
            return None

        data_aux = []
        x_ = []
        y_ = []

        H, W, _ = frame.shape

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,  # image to draw
                    hand_landmarks,  # model output
                    mp_hands.HAND_CONNECTIONS,  # hand connections
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    x = landmark.x
                    y = landmark.y
                    x_.append(x)
                    y_.append(y)

                width = max(x_) - min(x_)
                height = max(y_) - min(y_)

                for landmark in hand_landmarks.landmark:
                    x = (landmark.x - min(x_)) / width
                    y = (landmark.y - min(y_)) / height
                    data_aux.append(x)
                    data_aux.append(y)

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = prediction[0]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

        cv2.imshow('frame', frame)
        cv2.waitKey(1)


def translate(capture_device=0, model_dir=MODEL_DIR):
    """
    Takes the video feed from a given device and takes hands gestures and translate them to a corresponding label.
    :param capture_device: The video capture device.
    :param model_dir: The directory of the models we can load.
    """
    if not os.path.exists(model_dir):
        raise Exception("Model directory not found!")
    cap = cv2.VideoCapture(capture_device)
    while True:
        language = get_sign_language(cap)
        if language is None:
            break

        if not os.path.exists(model_dir + '/' + language + '.pickle'):
            ret, frame = cap.read()
            display_text(frame, f"{language} model not found!, Please enter a valid model in {model_dir}")
            cv2.waitKey(500)

        else:
            model = get_language_model(model_dir + '/' + language + '.pickle')
            translate_in_real_time(cap, model, language)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    translate()
