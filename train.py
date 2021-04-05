# %%
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras

DATA_PATH = 'dataset.json'

LEARNING_RATE = 0.0001
EPOCHS = 50
BATCH_SIZE = 32
SAVED_MODEL_PATH = "model.h5"

NUM_KEYWORDS = 50


# %%
def load_dataset(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)
    X = np.array([line["mfcc"] for line in data])
    y = np.array([line["target"] for line in data])
    print(X.shape, y.shape)
    return X, y

#%%
# load_dataset(DATA_PATH)


# %%
def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    X, y = load_dataset(data_path)
    # X = X.T
    print(X, y, X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train,
                                                                    test_size=test_validation)

    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print(X_train.shape, X_test.shape, X_validation.shape,
          y_train.shape, y_test.shape, y_validation.shape)

    return X_train, X_validation, X_test, y_train, y_validation, y_test


# %%
def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):
    # network
    model = keras.Sequential()

    # conv layer 1
    model.add(keras.layers.Conv2D(64, (3, 3), activation="relu",
                                  input_shape=input_shape,
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 2
    model.add(keras.layers.Conv2D(32, (3, 3), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3, 3), strides=(2, 2), padding="same"))

    # conv layer 3
    model.add(keras.layers.Conv2D(32, (2, 2), activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding="same"))

    # flattening the output
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # softmax
    model.add(keras.layers.Dense(NUM_KEYWORDS, activation="softmax"))

    # compiling the model
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimiser, loss=error, metrics=["accuracy"])

    # print model overview
    model.summary()
    return model


# %%
def main():
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(
        DATA_PATH)

    print(y_train.shape)

    # build the model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    print(f'type of input shape {input_shape}')
    model = build_model(input_shape, LEARNING_RATE)

    # train the model
    model.fit(X_train, y_train, epochs=EPOCHS,
              validation_data=(X_validation, y_validation))

    # evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}, test accuracy: {test_accuracy}")

    # saving the model
    model.save(SAVED_MODEL_PATH)


if __name__ == '__main__':
    main()
