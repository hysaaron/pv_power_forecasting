# %% [markdown]
# # 光伏发电预测
# 
# fork https://github.com/irutheu/LSTM-power-forecasting.git 仓库。
# 仓库中使用LSTM模型进行预测。

# %%
# import libs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

# model itself
from keras.models import Sequential
from keras.layers import LSTM, Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import GaussianNoise
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from keras.losses import mean_absolute_percentage_error
# from keras import backend as K
from keras.src import ops as K

from tqdm import trange

from models.tcn import TCN

# %%
df = pd.read_csv('./data/pvdaq_2012_2014_hourly.csv', header=0, infer_datetime_format=True, parse_dates=['Date-Time'], index_col=['Date-Time'])

# %%
# columns to use for forecasting
cols = ['ambient_temp', 'inverter_temp', 'module_temp', 'poa_irradiance', 
        'relative_humidity', 'wind_direction', 'wind_speed']
time_indexes = [df.index.hour, df.index.month]
# we will forecast dc power output
target = ['dc_power']

# %%
# array stacking
def create_sequence(df, cols, target):
  seqs = []
  for col in cols:
    seq = df[col].values.reshape((len(df[col]), 1))
    seqs.append(seq)
  for index in time_indexes:
    seq = index.values.reshape((len(df[col]), 1))
    seqs.append(seq)
  seq = df[target].values.reshape((len(df[target]), 1))
  for i in range(len(seq)):
    if seq[i] < 0:
      seq[i] = 0
  seqs.append(seq)
  dataset = np.hstack((seqs))  
  return dataset

dataset = (create_sequence(df, cols, target))

# %%
# single step multivariate sequence
def split_sequence(sequence, n_steps):
  X, y = list(), list()
  for i in range(len(sequence)):
    end_ix = i + n_steps
    # check if we are not beyond range
    if end_ix > len(sequence)-1:
      break
    seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix ,-1]
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)

# %%
def split_sequence_multi(sequence, n_steps, n_steps_out):
  X, y = list(), list()
  for i in range(len(sequence)):
    end_ix = i + n_steps
    out_ix = end_ix + n_steps_out
    # boundary check
    if out_ix > len(sequence):
      break
    seq_x, seq_y = sequence[i:end_ix, :], sequence[end_ix:out_ix, -1]
    X.append(seq_x)
    y.append(seq_y)
  return np.array(X), np.array(y)

# %%
# n_steps is amount of time steps per sample
# n_steps_out is the amount of time steps model has to forecast
n_steps, n_steps_out = 24, 6 # 24, 6
# number of features in each timestep
X, y = split_sequence_multi(dataset, n_steps, n_steps_out)
n_features = X.shape[2]
# shuffle datasets
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
# split datasets
num_val = 5000
num_test = 5000
train_X, train_y = X[:-(num_val+num_test),:], y[:-(num_val+num_test),:]
val_X, val_y = X[-(num_val+num_test):-num_test,:], y[-(num_val+num_test):-num_test,:]
test_X, test_y = X[-num_test:,:], y[-num_test:,:]
print(train_X.shape, val_X.shape, test_X.shape)

# %%
# Model
model = Sequential()

# model.add(GaussianNoise(0.5))

# LSTM
# model.add(LSTM(256, activation='tanh', return_sequences=True, input_shape=(n_steps, n_features)))
# model.add(BatchNormalization())
# model.add(LSTM(256, activation='tanh'))
# model.add(BatchNormalization())
# # model.add(Dense(64)) # , activation='relu'
# # model.add(BatchNormalization())
# model.add(PV_LSTM())

# Pure TCN
model.add(TCN(64, 7, dilations=[1, 2, 4, 8], use_weight_norm=False, return_sequences=False, dropout_rate=0.5, activation='relu'))

# Final Dense layer
model.add(Dense(n_steps_out))


# %%
# Training
model.compile(optimizer='adamw', loss='mae') # mse
# model = build_model()

history = model.fit(train_X, train_y, batch_size=32, epochs=20, validation_data=(val_X, val_y))
# A stateful recurrent model is one for which the internal states (memories) 
# obtained after processing a batch of samples are reused as initial states for the samples of the next batch

predictions = model.predict(test_X)

# %%
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Forecasting Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper right')
plt.savefig('./outputs/history.png')
plt.show()

# %%
# https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
def sMAPE(y_true, y_pred):
    #Symmetric mean absolute percentage error
    # return 100 * K.mean(K.abs(y_pred - y_true) / (K.abs(y_pred).numpy() + K.abs(y_true).numpy()), axis=-1)
    return K.sum(K.abs(y_pred - y_true)) / K.sum(K.abs(y_pred).numpy() + K.abs(y_true).numpy())

def calculate_PICP(y_true, y_pred, epsilon):
    # 区间置信度
    # 区间置信度表示为概率：P(|y_true - y_pred| <= epsilon) >= alpha，然而有限的观测数据无法统计出准确的概率。
    # 故使用预测区间覆盖率（prediction interval coverage probability, PICP）
    # 和平均预测区间宽度（mean prediction interval width, MPIW）来综合表示区间预测准确率。
    # 在确保 PICP 大于区间置信度（本课题设置为85%）的同时，MPIW 越窄准确度越高。
    c_i = K.abs(y_true - y_pred) <= epsilon
    return K.sum(c_i) / y_true.flatten().shape[0]

def calculate_MPIW(y_true, y_pred, alpha):
    # alpha 表示目标置信度概率，本课题设置为85%，代表预测值需有85%在置信区间宽度范围内。
    # 此时，宽度越窄表示预测越精准。
    # beta 表示实际置信度概率
    # gamma 表示宽度（应该用每组数据的宽度做平均，这里暂且简化）
    for interval in trange(1000, 10000):
      c_i = K.abs(y_true - y_pred) <= interval
      beta = K.sum(c_i) / y_true.flatten().shape[0]
      gamma = interval * 2.0 / 1000.0 # 单位改成 MW
      if beta >= alpha:
        return beta, gamma
      
    return -1, -1

# %%
predictions1 = model.predict(train_X)
mse = mean_squared_error(train_y, predictions1)
rmse = sqrt(mse)
mae = mean_absolute_error(train_y, predictions1)
#mape = mean_absolute_percentage_error(predictions1, test_y)
print(round(mse), round(rmse), round(mae))

# %%
mse = mean_squared_error(test_y, predictions)
rmse = sqrt(mse)
mae = mean_absolute_error(test_y, predictions)
# mape = mean_absolute_percentage_error(test_y, predictions)
smape = sMAPE(test_y, predictions)
r2 = r2_score(test_y, predictions)
# acc_epsilon = calculate_PICP(test_y, predictions, 3000.)
interval_confidence, confidence_width = calculate_MPIW(test_y, predictions, 0.85) # 本课题设置置信度为85%
# print(round(mse), round(rmse), round(mae))
print(f"MSE: {mse}")
print(f"rMSE: {rmse}")
print(f"MAE: {mae}")
print(f"sMAPE: {smape}")
print(f"R2 score: {r2}")
print(f"PICP: {interval_confidence}")
print(f"MPIW: {confidence_width}")

# %%
test_y[0], test_y[6], test_y[12], test_y[18], test_y[24]
predictions[0], predictions[6], predictions[12], predictions[18], predictions[24]

# %%
test_seq1 = []
pred_seq1 = []
for i in range(36):
  test_seq1 = np.concatenate((test_seq1, test_y[i*6]))
  pred_seq1 = np.concatenate((pred_seq1, predictions[i*6]))

# %%
import matplotlib.lines as mlines

blue_line = mlines.Line2D([], [], color='blue', marker='s',
                          markersize=5, label='label')
red_line = mlines.Line2D([], [], color='red', marker='p',
                          markersize=5, label='prediction')

#plt.legend(handles=[blue_line, red_line])

plt.figure(figsize=(18,10))
plt.plot(test_seq1, 'b-s')
plt.plot(pred_seq1, 'r--p')
plt.legend(handles=[blue_line, red_line])
plt.savefig('./outputs/figure.png')

# %%
test_seq = np.concatenate((test_y[0], test_y[6], test_y[12], test_y[18], test_y[24], test_y[30], test_y[36]))
pred_seq = np.concatenate((predictions[0], predictions[6], predictions[12], predictions[18], predictions[24], predictions[30], predictions[36]))
plt.plot(test_seq)
plt.plot(pred_seq)

# %%
from keras.utils import plot_model
   
### Build, Load, and Compile your model  
   
#  plot_model(model, to_file='model.png', show_layer_names=True)
plot_model(model, to_file='./outputs/model.png', show_shapes=True, show_layer_activations=True)
model.save("./outputs/model.keras")
