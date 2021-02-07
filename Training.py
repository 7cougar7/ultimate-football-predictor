import pandas as pd
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("TensorFlow Version: {}".format(tf.__version__))
    print("Eager execution: {}".format(tf.executing_eagerly()))

    df = pd.read_csv('finalized_db.csv')
    # target = df[['home_score', 'visiting_score', 'win_binary']]
    target = df[['win_binary']]
    df = df.drop([df.columns[0], 'home_score', 'visiting_score', 'win_binary'], axis=1)
    dataset = tf.data.Dataset.from_tensor_slices((df.values, target.values))
    # for item in dataset:
    #     print(item)
    # assert not np.any(np.isnan(df))
    # df = df.iloc[1:]
    # print(df)
    # for feet, targ in dataset.take(5):
    #     print('Features: {}, Target: {}'.format(feet, targ))

    train_dataset = dataset.shuffle(len(df)).batch(1)
    test_dataset = dataset
    model = get_compiled_model()
    # model.fit(train_dataset, epochs=1)
    # prediction = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    # prediction = np.reshape(prediction, (1, 14))
    EPOCHS = 2
    optimizer = tf.keras.optimizers.Adam()
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    for epoch in range(EPOCHS):
        for (x_train, y_train) in train_dataset:
            train_step(model, optimizer, x_train, y_train)
        # with train_summary_writer.as_default():
        #     tf.summary.scalar('loss', train_loss.result(), step=epoch)
        #     tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

        for (x_test, y_test) in test_dataset:
            test_step(model, x_test, y_test)
        # with test_summary_writer.as_default():
        #     tf.summary.scalar('loss', test_loss.result(), step=epoch)
        #     tf.summary.scalar('accuracy', test_accuracy.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_accuracy.result() * 100,
                              test_loss.result(),
                              test_accuracy.result() * 100))

        # Reset metrics every epoch
        train_loss.reset_states()
        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()

    # print(prediction, '<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    # print(model.predict(prediction), '<<<<<<<<<<<<<<<<<<<<<<<')
    # print(prediction.shape, '<<<<<<<<<<<<<<<')


def train_step(model, optimizer, x_train, y_train):
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('train_accuracy')
    loss_object = tf.keras.losses.MeanAbsoluteError()
    with tf.GradientTape() as tape:
        predictions = model(x_train)
        loss = loss_object(y_train, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    train_loss(loss)
    train_accuracy(y_train, predictions)


def test_step(model, x_test, y_test):
    test_loss = tf.keras.metrics.Mean('test_loss', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy('test_accuracy')
    loss_object = tf.keras.losses.MeanAbsoluteError()
    x_test = np.reshape(x_test, (1, 14))
    predictions = model(x_test)
    loss = loss_object(y_test, predictions)
    test_loss(loss)
    test_accuracy(y_test, predictions)


def get_compiled_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(14, activation=tf.nn.sigmoid, input_shape=(14,)),
        tf.keras.layers.Dense(12, activation=tf.nn.sigmoid),  # maybe reLU >:(
        # tf.keras.layers.Dense(3)
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MeanAbsoluteError(),
                  metrics=['accuracy'])
    return model


'''
    
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # this will probably change

def loss(model, x, y):
    y_ = model(x)
    return loss_object(y_true=y, y_pred=y_)

#l = loss(model, features, labels)  # i think features are inputs and labels are outputs. idk tbh
# https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_value, grads = grad(model, features, labels)
optimizer.apply_gradients(zip(grads, model.trainable_variables))


# The whole shabang

train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        loss_value, grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y, model(x))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

# Plotting the loss function over iterations for funsies
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)
plt.show()
'''

if __name__ == '__main__':
    main()
