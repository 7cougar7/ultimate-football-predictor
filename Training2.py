import pandas as pd
import tensorflow as tf
import numpy as np
import os


def main():
    model_version = 4
    df = pd.read_csv('all_inputs.csv')
    target = df[['home score', 'visiting score', 'Win Binary']]
    df = df.drop([df.columns[0], 'home score', 'visiting score', 'Win Binary'], axis=1)
    normalized_dataset = (df - df.min()) / (df.max() - df.min())

    n = 600
    validation_indices = np.random.choice(normalized_dataset.index, n, replace=False)
    validation_dataset = normalized_dataset.iloc[validation_indices]
    validation_output = target.iloc[validation_indices]
    training = normalized_dataset.drop(validation_indices)
    training_output = target.drop(validation_indices)

    validation_dataset = validation_dataset.reset_index()
    validation_output = validation_output.reset_index()
    training = training.reset_index()
    training_output = training_output.reset_index()

    validation_dataset = validation_dataset.drop(['index'], axis=1)
    validation_output = validation_output.drop(['index'], axis=1)
    training = training.drop(['index'], axis=1)
    training_output = training_output.drop(['index'], axis=1)

    x_train = normalized_dataset.values
    y_train = target.values

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    model = create_model()
    EPOCHS = 10000

    model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    # model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanSquaredError(),
    #               metrics=['accuracy'])

    # checkpoint_path = "models/Model_v" + str(model_version)
    # checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    # cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
    #                                                 save_weights_only=True,
    #                                                 verbose=1, save_freq=5)

    model.fit(x=training.values, y=training_output.values, epochs=EPOCHS,
              validation_data=(validation_dataset.values, validation_output.values),
              use_multiprocessing=True)  # , callbacks=[cp_callback])
    print(validation_output)
    print(model(validation_dataset.values))

    model.save('models/Model_v' + str(model_version))


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(16,)),
        tf.keras.layers.Dense(6, activation=tf.keras.activations.tanh),
        tf.keras.layers.Dense(6, activation=tf.keras.activations.tanh),
        # tf.keras.layers.Dense(6, activation=tf.nn.tanh),
        # tf.keras.layers.Dense(6, activation=tf.nn.tanh),
        # tf.keras.layers.Dense(10, activation=tf.nn.tanh),
        tf.keras.layers.Dense(3)
    ])
    return model


if __name__ == '__main__':
    main()
