import pandas as pd
import tensorflow as tf
import numpy as np

# This function normalizes the dataset using min max scaling.
# DO NOT CHANGE THIS CODE
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data

# This function is used to map the time series dataset into windows of
# features and respective targets, to prepare it for training and validation.
# The first element of the first window will be the first element of
# the dataset.
#
# Consecutive windows are constructed by shifting the starting position
# of the first window forward, one at a time (indicated by shift=1).
#
# For a window of n_past number of observations of the time
# indexed variable in the dataset, the target for the window is the next
# n_future number of observations of the variable, after the
# end of the window.

# DO NOT CHANGE THIS.
def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)

def solution_model():
    # DO NOT CHANGE THIS CODE
    # Reads the dataset.
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features of future time steps.
    N_FEATURES = 1  # DO NOT CHANGE THIS

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.8)  # DO NOT CHANGE THIS
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    # DO NOT CHANGE THIS CODE
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    # DO NOT CHANGE BATCH_SIZE IF YOU ARE USING STATEFUL LSTM/RNN/GRU.
    # THE TEST WILL FAIL TO GRADE YOUR SCORE IN SUCH CASES.
    # In other cases, it is advised not to change the batch size since it
    # might affect your final scores. While setting it to a lower size
    # might not do any harm, higher sizes might affect your scores.
    BATCH_SIZE = 32  # ADVISED NOT TO CHANGE THIS

    # DO NOT CHANGE N_PAST, N_FUTURE, SHIFT. The tests will fail to run
    # on the server.
    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 10  # DO NOT CHANGE THIS

    # Number of future time steps which are to be predicted.
    N_FUTURE = 10  # DO NOT CHANGE THIS

    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1  # DO NOT CHANGE THIS

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    # Code to define your model.
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(N_PAST, N_FEATURES)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(N_FUTURE * N_FEATURES, activation='sigmoid'),
        tf.keras.layers.Reshape((N_FUTURE, N_FEATURES))
    ])

    # Code to train and compile the model
    optimizer = tf.keras.optimizers.Adam()
    model.compile(optimizer=optimizer, loss='mae')

    # Add EarlyStopping callback
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    model.fit(train_set, validation_data=valid_set, epochs=1000, callbacks=[early_stop])

    return model

# Note that you'll need to save your model as a .h5 like this.
# When you press the Submit and Test button, your saved .h5 model will
# be sent to the testing infrastructure for scoring
# and the score will be returned to you.

if __name__ == '__main__':
    model = solution_model()
    model.save("c5q12.h5")

    # This code is used for testing your model locally
    def model_forecast(model, series, window_size, batch_size):
        ds = tf.data.Dataset.from_tensor_slices(series)
        ds = ds.window(window_size, shift=1, drop_remainder=True)
        ds = ds.flat_map(lambda w: w.batch(window_size))
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(1)
        forecast = model.predict(ds)
        return forecast

    # PASS THE NORMALIZED data IN THE FOLLOWING CODE
    df = pd.read_csv('Weekly_U.S.Diesel_Retail_Prices.csv',
                     infer_datetime_format=True, index_col='Week of', header=0)
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    SPLIT_TIME = int(len(data) * 0.8)
    N_PAST = 10
    BATCH_SIZE = 32

    x_valid = data[SPLIT_TIME:]

    rnn_forecast = model_forecast(model, data, N_PAST, BATCH_SIZE)
    rnn_forecast = rnn_forecast[SPLIT_TIME - N_PAST:-1, 0, 0]

    x_valid = np.squeeze(x_valid[:rnn_forecast.shape[0]])
    result = tf.keras.metrics.mean_absolute_error(x_valid, rnn_forecast).numpy()
    print("Validation MAE:", result)
