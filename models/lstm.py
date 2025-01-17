from keras.models import Sequential
from keras.layers import LSTM, Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization


class PV_LSTM(Layer):
  """Creates a LSTM layer.
  
  """
  model.add(LSTM(256, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
  model.add(BatchNormalization())
  model.add(LSTM(256, activation='tanh'))
  model.add(BatchNormalization())
  # model.add(Dense(64)) # , activation='relu'