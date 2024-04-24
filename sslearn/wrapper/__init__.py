"""
Summary of module `sslearn.wrapper`:

This module contains classes to train semi-supervised learning algorithms using a wrapper approach.

Self-Training Algorithms
------------------------
1. SelfTraining : Self-training algorithm.
2. Setred : Self-training with redundancy reduction.

Co-Training Algorithms
-----------------------
1. CoTraining : Co-training
2. CoTrainingByCommittee : Co-training by committee
3. DemocraticCoLearning : Democratic co-learning
4. Rasco : Random subspace co-training
5. RelRasco : Relevant random subspace co-training
6. CoForest : Co-Forest
7. TriTraining : Tri-training
8. DeTriTraining : Data Editing Tri-training
9. WiWTriTraining : Who-Is-Who Tri-training

All doc
----
"""

from ._co import (CoForest, CoTraining, CoTrainingByCommittee,
                  DemocraticCoLearning, Rasco, RelRasco)
from ._self import SelfTraining, Setred
from ._tritraining import DeTriTraining, TriTraining, WiWTriTraining

__all__ = ["SelfTraining", "CoTrainingByCommittee", "Rasco", "RelRasco", "TriTraining", "WiWTriTraining",
           "CoTraining", "DeTriTraining", "DemocraticCoLearning", "Setred", "CoForest"]
