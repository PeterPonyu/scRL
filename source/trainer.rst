Trainer
========

Four learners are available through ``scRL.trainer(algo, ...)``: tabular Q-learning (``'TabularQ'``), an
actor-critic (``'ActorCritic'``), double deep Q-learning (``'DDQN'``) and a soft actor-critic (``'SAC'``).
The soft actor-critic learns two Q networks, a policy that keeps some randomness and a temperature that sets
how much: ``entropy_frac`` is the target entropy of the policy as a fraction of the largest possible one
(default 0.3). It learns from a replay buffer like DDQN, usually needs fewer episodes than the other learners
(a few thousand) but more time per episode. ``trainer.state_values()`` gives the value of every grid point
for any learner.

.. autosummary::
    :toctree: doc
    :nosignatures:
    
    scRL.Trainer.trainer