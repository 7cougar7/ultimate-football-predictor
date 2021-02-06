import pandas as pd
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    # print("TensorFlow Version: {}".format(tf.__version__))
    # print("Eager execution: {}".format(tf.executing_eagerly()))

    df = pd.read_csv('finalized_db.csv')
    # target = df[['home_score', 'visiting_score', 'win_binary']]
    target = df[['win_binary']]
    df = df.drop([df.columns[0], 'home_score', 'visiting_score', 'win_binary'], axis=1)
    random_dataset = pd.DataFrame(np.random.randint(0, 500, size=(267, 14)))
    my_dataset = pd.DataFrame(np.array([[1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]]))
    my_output = pd.DataFrame(np.array([[1], [0]]))
    my_normalized = (my_dataset - my_dataset.min()) / (my_dataset.max() - my_dataset.min())
    random_normalized_dataset = (random_dataset - random_dataset.min()) / (random_dataset.max() - random_dataset.min())
    normalized_dataset = (df - df.min()) / (df.max() - df.min())
    # print(random_normalized_dataset.values)
    # print()
    # print(normalized_dataset.values)
    # print(normalized_random_dataset)
    random_output = pd.DataFrame(np.random.randint(0, 2, size=(267, 1)))
    # train_dataset = dataset.shuffle(len(df)).batch(1)
    test_dataset = df
    # train_dataset = tf.data.Dataset.from_tensor_slices((random_dataset.values, random_output.values))
    # x_train = my_normalized.values
    # y_train = my_output.values
    # x_train = random_normalized_dataset.values
    # y_train = random_output.values
    x_train = normalized_dataset.values
    y_train = target.values
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
    predictions = model(x_train)
    # print(model.loss(y_true=target.values, y_pred=predictions))
    model.fit(x=x_train, y=y_train, epochs=30, validation_data=(df.values, target.values), verbose=1)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.sigmoid, input_shape=(14,)),
        tf.keras.layers.Dense(10, activation=tf.nn.tanh),
        # tf.keras.layers.Dense(3)
        tf.keras.layers.Dense(1, activation=tf.nn.tanh)
    ])
    return model


if __name__ == '__main__':
    main()
