import tensorflow as tf
import pandas as pd

def main():

    df = pd.read_csv('finalized_db.csv')
    target = df[['home_score', 'visiting_score', 'win_binary']]
    df = df.drop([df.columns[0], 'home_score', 'visiting_score', 'win_binary'], axis=1)
    normalized_dataset = (df - df.min()) / (df.max() - df.min())
    x_train = normalized_dataset.values
    y_train = target.values

    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    checkpoint_path = "models/Model_v2"
    model.load_weights(checkpoint_path)

    # Re-evaluate the model
    loss, acc = model.evaluate(x_train, y_train, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation=tf.nn.tanh, input_shape=(14,)),
        tf.keras.layers.Dense(10, activation=tf.nn.tanh),
        tf.keras.layers.Dense(10, activation=tf.nn.tanh),
        tf.keras.layers.Dense(10, activation=tf.nn.tanh),
        tf.keras.layers.Dense(3)
    ])
    return model


if __name__ == "__main__":
    main()
