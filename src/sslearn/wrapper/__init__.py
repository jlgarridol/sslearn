"""
Summary of module `sslearn.wrapper`:

This module contains classes to train semi-supervised learning algorithms using a wrapper approach.

## Self-Training Algorithms

* [SelfTraining](#SelfTraining): 
Self-training algorithm.
* [Setred](#Setred):
Self-training with redundancy reduction.

## Co-Training Algorithms

* [CoTraining](#CoTraining):
Co-training
* [CoTrainingByCommittee](#CoTrainingByCommittee):
Co-training by committee
* [DemocraticCoLearning](#DemocraticCoLearning):
Democratic co-learning
* [Rasco](#Rasco):
Random subspace co-training
* [RelRasco](#RelRasco):
Relevant random subspace co-training
* [CoForest](#CoForest):
Co-Forest
* [TriTraining](#TriTraining):
Tri-training
* [DeTriTraining](#DeTriTraining):
Data Editing Tri-training

"""

from ._co import (CoForest, CoTraining, CoTrainingByCommittee,
                  DemocraticCoLearning, Rasco, RelRasco)
from ._self import SelfTraining, Setred
from ._tritraining import DeTriTraining, TriTraining

__all__ = ["SelfTraining", "Setred", "CoTraining", "CoTrainingByCommittee",
           "DemocraticCoLearning", "Rasco", "RelRasco", "CoForest",
           "TriTraining", "DeTriTraining"]
