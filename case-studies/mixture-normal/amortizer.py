import os, sys
# for accessing src, stan, etc.
sys.path.append(os.path.abspath(os.path.join("../..")))


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from bayesflow.trainers import Trainer
from bayesflow.amortizers import AmortizedPosterior
from bayesflow.networks import InvertibleNetwork
from bayesflow.summary_networks import DeepSet, HierarchicalNetwork

from src.networks import AmortizedMixture, AmortizedPosteriorMixture
from src.models.MixtureNormal import model, configurator

local_summary_net = DeepSet(summary_dim=2)
global_summary_net = DeepSet(summary_dim=20,
                             num_dense_s1=4, num_dense_s2=4, num_dense_s3=3, num_equiv=3)
classification_net = Sequential([Dense(64, activation='relu') for _ in range(12)])
classification_net.add(Dense(3))


amortizer = AmortizedPosteriorMixture(
    amortized_posterior=AmortizedPosterior(
        inference_net=InvertibleNetwork(num_params=5, num_coupling_layers=14, coupling_design="spline"),
        summary_net=HierarchicalNetwork([local_summary_net, global_summary_net]),
        summary_loss_fun="MMD"
    ),
    amortized_mixture=AmortizedMixture(
        inference_net=classification_net, 
        local_summary_net=local_summary_net
    )
)

trainer = Trainer(
    amortizer=amortizer, 
    generative_model=model, 
    configurator=configurator, 
    checkpoint_path="checkpoints/amortizer"
)