# This is a Machine Learning simple proyect that shows how Neural Networks (NNs) try to approximate any function.
# We can generate synthetic data (with and without noise) and understand how neural networks can approximate this by fitting the data.
# Depending on the activation function, could the neural newtork approximate the functions?

# In this project, we are going to use the most basic architecture of NNs: the Multilayer Perceptron (MLP)


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence TensorFlow's logs (0 = ALL, 1 = Filter INFO, 2 = Filter INFO/WARN, 3 = Filter ALL)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Deactivate oneDNN optimization warnings 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Flatten, Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import (
    HeNormal, HeUniform, 
    GlorotNormal, GlorotUniform, 
    LecunNormal, RandomNormal, 
    Zeros, Ones
)


gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"using GPU?: Yes ({len(gpus)} detected")
    for gpu in gpus:
        print(f"  - {gpu.name}")
else:
    print("using GPU?: No (CPU instead)")

# CONTROL VARIABLES
initial_plot = False


# two different partitions, two different datasets
## 1. Ordered partition

x_full  = np.linspace(0, 120000, 100000) # 1-Dimensional VECTOR ---> dimension: (N,)
y_full  = np.log1p(x_full)               # 1-Dimensional VECTOR ---> dimension: (N,)

n = len(x_full)
idx_train = int(n * 0.6)
idx_val   = int(n * 0.8)

x_train_1 = x_full[:idx_train].reshape(-1, 1)
y_train_1 = y_full[:idx_train].reshape(-1, 1)

x_val_1   = x_full[idx_train:idx_val].reshape(-1, 1)
y_val_1   = y_full[idx_train:idx_val].reshape(-1, 1)

x_test_1  = x_full[idx_val:].reshape(-1, 1)
y_test_1  = y_full[idx_val:].reshape(-1, 1)



if initial_plot:
    plt.figure()
    plt.plot(x_train_1, y_train_1, '.', color='blue', label='train')
    plt.plot(x_test_1, y_test_1, '.', color='red', label='test')
    plt.title('Full dataset representation')
    plt.ylabel('y')
    plt.xlabel('x')
    plt.legend()
    plt.show()



### Model creation (Keras functional API)

def build_model(x_input, LR=0.01, num_layers=5, neurons_0=256, reduce_factor=2, input_dim=1):
    # -------------------------------------------------------------------
    # INITIALIZATION OPTIONS (Uncomment the one you want to test)
    # -------------------------------------------------------------------
    
    # --- 1. HE FAMILY (Recommended for 'relu' / 'leaky_relu') ---
    kernel_init = HeNormal(seed=42)                  # Gold standard for ReLU (Gaussian/Normal)
    # kernel_init = HeUniform(seed=42)                 # Uniform variant for ReLU
    # kernel_init = 'he_normal'                        # Direct string (default config)
    # kernel_init = 'he_uniform'                       # Direct string

    # --- 2. GLOROT / XAVIER FAMILY (Recommended for 'tanh' / 'sigmoid' / 'linear') ---
    # kernel_init = GlorotNormal(seed=42)             # Glorot Gaussian
    # kernel_init = GlorotUniform(seed=42)            # Glorot Uniform (Keras default if unspecified)
    # kernel_init = 'glorot_normal'                   # Direct string
    # kernel_init = 'glorot_uniform'                  # Direct string

    # --- 3. LECUN FAMILY (Recommended for 'selu') ---
    # kernel_init = LecunNormal(seed=42)              # Recommended when using SELU activation
    # kernel_init = 'lecun_normal'                    # Direct string

    # --- 4. OTHER BASIC INITIALIZERS ---
    # kernel_init = RandomNormal(mean=0.0, stddev=0.05, seed=42) # Simple Gaussian (highly sensitive to stddev)
    # kernel_init = Zeros()                           # NEVER use for weights! Causes symmetry problem
    # kernel_init = Ones()                            # NEVER use for weights!

    inputs = Input(shape=(input_dim,)) 
    x = Flatten()(inputs)

    for layer in range(num_layers):
        if layer == 0:
            neurons = neurons_0
        else:
            neurons = neurons // reduce_factor
        x = Dense(neurons, activation='relu')(x)


    outputs = Dense(1, activation='linear')(x)    

    model = Model(inputs=inputs, outputs=outputs, name="functional_model")
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss='mean_absolute_error',
        metrics=['mean_absolute_error']
    )

    return model



custom_callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
    ]

model_0 = build_model(x_train_1) # default hyperparameters, not the optimal ones



model_0.fit(
        x_train_1, y_train_1,
        validation_data=(x_val_1, y_val_1),
        epochs=100,
        batch_size=64,
        callbacks=custom_callbacks,
        verbose=2) # 0 = no log; 1 = full verbosity; 2 = only 1 line per epoch











