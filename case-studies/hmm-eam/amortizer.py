import os, sys
# for accessing src, stan, etc.
sys.path.append(os.path.abspath(os.path.join("../..")))


import tensorflow as tf
import bayesflow as bf
import numpy as np


from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

from bayesflow.trainers import Trainer
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork, SequenceNetwork, TimeSeriesTransformer
from bayesflow.summary_networks import DeepSet, HierarchicalNetwork

from src.networks import AmortizedSmoothing, AmortizedPosteriorMixture, Classifier
from src.models.HmmEam import model, configurator, constrain_parameters, constrained_parameter_names


#local_summary_net = DeepSet(summary_dim=2)
classification_net = Sequential([
    LSTM(units=32, return_sequences=True), 
    Classifier(n_classes=2, n_units=[16, 8, 4])
])

amortizer = AmortizedPosteriorMixture(
    amortized_posterior=AmortizedPosterior(
        inference_net=InvertibleNetwork(num_params=8, num_coupling_layers=10, coupling_design="spline"),
        summary_net=SequenceNetwork(summary_dim=24, bidirectional=True),#TimeSeriesTransformer(summary_dim=24, input_dim=3),
        summary_loss_fun="MMD"
    ),
    amortized_mixture=AmortizedSmoothing(
        forward_net=classification_net, 
        backward_net=classification_net
    )
)

trainer = Trainer(amortizer=amortizer, generative_model=model, configurator=configurator, checkpoint_path="checkpoints/amortizer")