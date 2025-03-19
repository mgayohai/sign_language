import os
import cv2

DATA_DIR = "./Data"
DATASET_SIZE = 100


def setup_directory(data_dir):
    """
    Creates a directory for our data collection.
    :param data_dir: The directory path.
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)


def display_text(frame, text, position=50, color=(0, 0, 0)):
    """
    Displays a text on the frame of the captured video.
    :param frame: The frame of the video capture object.
    :param text: The inserted text to be displayed on the screen.
    :param position: The text position on the y-axis of the frame.
    :param color: The colour of the inserted text.
    """
    font_scale = frame.shape[1] / 800  # Scale font relative to width
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, 3)
    text_x = max(10, (frame.shape[1] - text_size[0]) // 2)
    cv2.putText(frame, text, (text_x, position), cv2.FONT_HERSHEY_TRIPLEX, font_scale, color, 3, cv2.LINE_AA)
    cv2.imshow('frame', frame)


def get_class_name(cap):
    """
    Gets the class name we want to add to our dataset.
    :param cap: The video capture object.
    :return: The new class' name.
    """
    class_name = ''

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display_text(frame, f"Please enter data classification: {class_name}")

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Esc key
            return None
        elif key == 13:  # Enter key
            if class_name == '':
                continue
            return class_name
        elif key == 8:  # Backspace key
            class_name = class_name[:-1]
        elif key != 255:
            class_name += chr(key)


def capture_class(cap, class_name, data_dir=DATA_DIR, dataset_size=DATASET_SIZE):
    """
    Captures images from the video capture object and saves them in the specified directory.
    :param cap: The video capture object.
    :param class_name: The class name we want to add to our dataset.
    :param data_dir: The directory for the data.
    :param dataset_size: Amount of images to be captured into our dataset.
    """
    setup_directory(os.path.join(data_dir, str(class_name)))  # Create directory for the class

    print('Collecting data for class {}'.format(class_name))

    while True:  # Wait until target is ready
        ret, frame = cap.read()
        display_text(frame, 'Press "R" When Ready', 50)
        if cv2.waitKey(25) == ord('r'):
            break

    for counter in range(dataset_size):  # Capture the images to the dataset
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)
        cv2.imwrite(os.path.join(data_dir, str(class_name), '{}.jpg'.format(counter)), frame)


def collect_data(capture_device=0, data_dir=DATA_DIR, dataset_size=DATASET_SIZE):
    """
    Creates a dataset of images for specified classes taken from the given capture device.
    :param capture_device: The capture device's integer.
    :param data_dir: The directory path to store the dataset
    :param dataset_size: The amount of images to be taken for each class in the dataset.
    """
    setup_directory(DATA_DIR)
    cap = cv2.VideoCapture(capture_device)
    while True:
        class_name = get_class_name(cap)
        if class_name is None:
            break
        capture_class(cap, class_name, data_dir, dataset_size)

    cap.release()
    cv2.destroyAllWindows()