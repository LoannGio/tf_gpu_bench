import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical
from optparse import OptionParser
import sys
import os
import json
import time
import socket

from displayResults import displayResults

parser = OptionParser()
parser.add_option("--gpu", dest="gpu", default="0")
parser.add_option("--gpu_name", dest="gpu_name")
parser.add_option("--host", dest="host_name")

(options, args) = parser.parse_args(sys.argv)
print(options)


if(options.gpu_name is None):
    print("Please provide GPU name: --gpu_name")
    exit()
    #options.gpu_name = "DEBUGGING"

if(options.host_name is None):
    options.host_name = socket.gethostname()


# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

os.environ["CUDA_VISIBLE_DEVICES"] = options.gpu

batch_size = 16

def handmade_lenet():
    input = Input(shape=(28, 28))
    x = Reshape((28, 28, 1))(input)
    x = Conv2D(6, (5, 5), activation="relu")(x)
    x = MaxPool2D()(x)
    x = Conv2D(12, (5, 5), activation="relu")(x)
    x = MaxPool2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dense(256, activation="relu")(x)
    x = Dense(10, activation="softmax")(x)
    return tf.keras.Model(input, x)


train_set, val_set = tf.keras.datasets.mnist.load_data(path="mnist.npz")
model = handmade_lenet()

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "categorical_crossentropy"])


start_timestamp = round(time.time(), 1)
train_history = model.fit(
    x=train_set[0],
    y=to_categorical(train_set[1]),
    batch_size=batch_size,
    epochs=50,
    validation_data=(val_set[0], to_categorical(val_set[1]))
)
end_timestamp = round(time.time(), 1)
elapsed_time = round(end_timestamp - start_timestamp, 1)

print(train_history.history.keys())
saveInfos = {
    "host": options.host_name,
    "GPU": options.gpu_name,
    "best_acc": max(train_history.history["val_accuracy"]),
    "best_crossentropy": max(train_history.history["val_categorical_crossentropy"]),
    "performance_time": str(elapsed_time) + " seconds"
    }

with open("res/result_"+str(int(start_timestamp))+".json", "w") as f:
    json.dump(saveInfos, f)
    f.close()

displayResults("./res", "./plots")