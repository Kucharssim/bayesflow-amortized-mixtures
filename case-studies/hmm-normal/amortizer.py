import os, sys
# for accessing src, stan, etc.
sys.path.append(os.path.abspath(os.path.join("../..")))


from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

from bayesflow.trainers import Trainer
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork, SequenceNetwork, DeepSet, HierarchicalNetwork

from src.networks import AmortizedSmoothing, AmortizedPosteriorMixture
from src.models.HmmNormal import model, configurator


local_summary_net = DeepSet(summary_dim=2)
classification_net = Sequential([
    LSTM(units=32, return_sequences=True), 
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense( 8, activation='relu'),
    Dense( 4, activation='relu'),
    Dense( 2)
])

amortizer = AmortizedPosteriorMixture(
    amortized_posterior=AmortizedPosterior(
        inference_net=InvertibleNetwork(num_params=4, num_coupling_layers=10, coupling_design="spline"),
        summary_net=HierarchicalNetwork([local_summary_net, SequenceNetwork(summary_dim=16, bidirectional=True)]),
        summary_loss_fun="MMD"
    ),
    amortized_mixture=AmortizedSmoothing(
        forward_net=classification_net, 
        backward_net=classification_net, 
        local_summary_net=local_summary_net
    )
)

trainer = Trainer(amortizer=amortizer, generative_model=model, configurator=configurator, checkpoint_path="checkpoints/amortizer")