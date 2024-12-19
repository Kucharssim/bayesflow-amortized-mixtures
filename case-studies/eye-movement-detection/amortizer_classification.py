import os, sys
# for accessing src, stan, etc.
sys.path.append(os.path.abspath(os.path.join("../..")))


from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from bayesflow.trainers import Trainer
from bayesflow.networks import TimeSeriesTransformer, SequenceNetwork


from src.networks import AmortizedSmoothing
from src.models.EyeMovements import model, configurator_mixture


classification_net = Sequential([
    LSTM(units=32, return_sequences=True), 
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense( 8, activation='relu'),
    Dense( 4, activation='relu'),
    Dense(2)
])

amortizer = AmortizedSmoothing(
    forward_net=classification_net, 
    backward_net=classification_net,
    global_summary_net=SequenceNetwork(summary_dim=14, bidirectional=True, lstm_units=8))#TimeSeriesTransformer(input_dim=(3), summary_dim=7, bidirectional=False))


trainer = Trainer(amortizer=amortizer, generative_model=model, configurator=configurator_mixture, checkpoint_path="checkpoints/amortizer_classification")