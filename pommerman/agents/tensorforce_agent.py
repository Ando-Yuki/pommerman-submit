"""
A Work-In-Progress agent using Tensorforce
"""
import os
import numpy as np
from . import BaseAgent
from .. import characters


class TensorForceAgent(BaseAgent):
    """The TensorForceAgent. Acts through the algorith, not here."""

    def __init__(self, character=characters.Bomber, algorithm='ppo',checkpoint='models/checkpoint'):
        super(TensorForceAgent, self).__init__(character)
        self.algorithm = algorithm
        self.checkpoint = checkpoint
        self.agent = None
        print(checkpoint)

    def act(self, obs, action_space):
        """This agent has its own way of inducing actions. See train_with_tensorforce."""
        return None

    def initialize(self, env):
        from gym import spaces
        from tensorforce.agents import PPOAgent

        if self.algorithm == "ppo":
            if type(env.action_space) == spaces.Tuple:
                actions = {
                    str(num): {
                        'type': int,
                        'num_actions': space.n
                    }
                    for num, space in enumerate(env.action_space.spaces)
                }
            else:
                actions = dict(type='int', num_actions=env.action_space.n)


            self.agent = PPOAgent(
                states=dict(type='float', shape=env.observation_space.shape),
                actions=actions,
                network=[
                    dict(type='dense', size=64),
                    dict(type='dense', size=64)
                ],
                batching_capacity=1000,
                step_optimizer=dict(type='adam', learning_rate=1e-4))#1e-4

            self.restore_model_if_exists(self.checkpoint)
        return self.agent

    def restore_model_if_exists(self, checkpoint):
        if os.path.isfile(checkpoint):
            pardir = os.path.abspath(os.path.join(checkpoint, os.pardir))
            self.agent.restore_model(pardir)
            print("tensorforce model '{}' restored.".format(pardir))
