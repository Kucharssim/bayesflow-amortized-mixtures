import os, sys
# for accessing src, stan, etc.
sys.path.append(os.path.abspath(os.path.join("../..")))

import tensorflow as tf
from tensorflow.keras import layers, models
from bayesflow.trainers import Trainer
from bayesflow.amortizers import AmortizedPointEstimator
from bayesflow.helper_networks import ConfigurableMLP

from src.models.EyeMovements2 import model, configurator_posterior

def build_summary_net(input_shape):
    inputs = tf.keras.Input(shape=input_shape)

    # Temporal feature extraction
    x = layers.Conv1D(filters=32, kernel_size=5, strides=2, activation='relu')(inputs)
    x = layers.Conv1D(filters=64, kernel_size=3, dilation_rate=2, activation='relu')(x)
    
    # Bidirectional LSTM for temporal statistics
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    # Self-attention mechanism
    attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)
    x = attention(x, x)

    # Fully connected layers
    outputs = layers.Bidirectional(layers.LSTM(32))(x)
    # outputs = layers.Dense(64, activation='relu')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

summary_net = build_summary_net((None, 3))
amortizer = AmortizedPointEstimator(
    inference_net=ConfigurableMLP(input_dim=64, output_dim=4, dropout_rate=0.0),
    summary_net=summary_net,
    norm_ord=2
)

trainer = Trainer(amortizer=amortizer, generative_model=model, configurator=configurator_posterior, checkpoint_path="checkpoints/amortizer_point_estimate")
