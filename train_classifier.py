import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collect_data import setup_directory

DATA_PICKLE_PATH = 'data.pickle'
MODEL_DIR = './Models'


def load_dataset(data_pickle_path):
    """
    Given a pickle dataset, loads it in order to train the model
    :param data_pickle_path: The path to the dataset
    :return: np array of the dataset and np array of their labels
    """
    with open(data_pickle_path, 'rb') as f:
        data_dict = pickle.load(f)

    data = []
    labels_cleaned = []

    for i in range(len(data_dict['data'])):
        if len(data_dict['data'][i]) == 42:  # 21 landmarks * 2 (x and y)
            data.append(data_dict['data'][i])
            labels_cleaned.append(data_dict['labels'][i])
        else:
            print(f"Skipping sample {i}: invalid length {len(data_dict['data'][i])}")

    return np.asarray(data), np.asarray(labels_cleaned)


def save_model(model_name, model):
    """
    Saves the trained model.
    :param model_name: The name the model will be saved under.
    :param model: The trained model
    """
    setup_directory(MODEL_DIR)
    f = open(MODEL_DIR + '/' + model_name + '.pickle', 'wb')
    pickle.dump({'model': model}, f)
    f.close()


def train_model(data, labels):
    """
    Trains the model to recognise the provided data and labels
    :param data: The data representing the hand positions.
    :param labels: The corresponding labels to the data.
    :return: The trainedmodel
    """
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    score = accuracy_score(y_predict, y_test)
    print("{}%of samples were classified correctly".format(score * 100))
    return model


def create_model(model_name, data_pickle_path=DATA_PICKLE_PATH):
    """
    Creates a model that predict sign language letters based on a given dataset.
    :param model_name: The name the model is saved under
    :param data_pickle_path: The path to the dataset.
    """
    data, labels = load_dataset(data_pickle_path)
    model = train_model(data, labels)
    save_model(model_name, model)

# create_model("asl")
