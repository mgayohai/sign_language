import os
import cv2

DATA_DIR = "./data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

dataset_size = 100
cap = cv2.VideoCapture(0)
class_name = ""
waiting_for_input = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame width dynamically
    frame_height, frame_width, _ = frame.shape
    font_scale = frame_width / 800  # Scale font relative to width

    if waiting_for_input:
        text = f"Please enter data classification: {class_name}"
    else:
        text = f'Class: {class_name}'

    # Calculate text size and center it horizontally
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_TRIPLEX, font_scale, 3)
    text_x = max(10, (frame_width - text_width) // 2)  # Ensure it stays within bounds

    cv2.putText(frame, text, (text_x, 50), cv2.FONT_HERSHEY_TRIPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.resizeWindow('frame', frame_width, frame_height)  # Ensure window scales with frame size
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # Esc key
        break
    elif key == 13:  # Enter key
        waiting_for_input = False
    elif waiting_for_input and key == 8:  # Backspace key
        class_name = class_name[:-1]
    elif waiting_for_input and key != 255:
        class_name += chr(key)

    if not waiting_for_input:
        class_dir = os.path.join(DATA_DIR, str(class_name))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print('Collecting data for class {}'.format(class_name))

        done = False
        while True:
            ret, frame = cap.read()
            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                break

        counter = 0
        while counter < dataset_size:
            ret, frame = cap.read()
            cv2.imshow('frame', frame)
            cv2.waitKey(25)
            cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
            counter += 1
        waiting_for_input = True
        class_name = ""
        key = ""

cap.release()
cv2.destroyAllWindows()
