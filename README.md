This repository contains code for analyzing videos of Lake Malawi male cichlids building sand bowers to attract female mates in naturalistic environments. Using Hidden Markov Models (HMMs) to identify long-lasting changes in pixel color followed by spatial, distance based clustering using dbSCAN, individual sand manipulation events are identified, including scoops and spits using the fishes mouth along with other sand changes caused by fins and the body.

testScript.py

An example of how the analysis can be run.

VideoClassification.py

Master script that runs a 3d convolutional neural network to classify video clips. This script runs 1) Training of the neural network from scratch. 2) Applying a pre_trained neural network to unobserved video clips. 

Arguments for this script include 


VideoClassification.yaml

Anaconda environment for running this repository

