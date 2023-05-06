import random
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, float64, nn, return_types
from pysc2.env.environment import StepType, TimeStep
import pysc2
from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np
import collections

SCREEN_SIZE = 64

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 1
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


class Experience(collections.namedtuple(
        'Experience', ['step_type', 'reward', 'action', 'observation'])):

    def first(self):
        return self.step_type is StepType.FIRST

    def mid(self):
        return self.step_type is StepType.MID

    def last(self):
        return self.step_type is StepType.LAST


class Model(nn.Module):
    def __init__(self, obs_size, available_actions):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(obs_size[0], 48, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(48, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(obs_size)
        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, available_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, obs_size):
        obs_size = [27, 64, 64]
        o = self.conv(torch.zeros(1, *obs_size))
        return int(np.prod(o.size()))

    def forward(self, screen):
        fx = screen / 256.0
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


def unpack_batch(batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.observation, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        if not exp.step_type == StepType.LAST:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.step_type, copy=False))
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    # actions_t = torch.LongTensor(actions).to(device)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_vals_v = net(states_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, -1]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions, ref_vals_v


class DefaultAgent(base_agent.BaseAgent):
    def __init__(self, net: Model):
        super(DefaultAgent, self).__init__()
        self.net = net
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.26, eps=1e-3)
        self.exp_source = []
        self.last_action = [0, []]
        self.GAMMA = 0.99

    def setup(self, obs_spec, action_spec):
        super(DefaultAgent, self).setup(obs_spec, action_spec)

    def can_do(self, obs, action):
        print(obs.observation['available_actions'])
        return action in obs.observation['available_actions']

    def step(self, obs: TimeStep):
        super(DefaultAgent, self).step(obs)
        self.exp_source.append(Experience(
            obs.step_type, obs.reward, self.last_action, torch.FloatTensor(obs.observation['feature_screen'])))
        states_v, actions_t, vals_ref_v = unpack_batch(
            self.exp_source, self.net)
        self.exp_source.clear()

        self.optimizer.zero_grad()
        logits_v, value_v = self.net(states_v)

        loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

        log_prob_v = F.log_softmax(logits_v, dim=1)
        adv_v = vals_ref_v - value_v.squeeze(-1).detach()
        log_prob_actions_v = adv_v * \
            log_prob_v[range(BATCH_SIZE), actions_t[0][0]]
        loss_policy_v = -log_prob_actions_v.mean()

        prob_v = F.softmax(logits_v, dim=1)
        entropy_loss_v = ENTROPY_BETA * \
            (prob_v * log_prob_v).sum(dim=1).mean()

        # calculate policy gradients only
        loss_policy_v.backward(retain_graph=True)
        grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
                                for p in self.net.parameters()
                                if p.grad is not None])

        # apply entropy and value gradients
        loss_v = entropy_loss_v + loss_value_v
        loss_v.backward()
        self.optimizer.step()
        # get full loss
        loss_v += loss_policy_v
        action = random.choice(actions_t)
        print(action)
        if self.can_do(obs, action[0]) and np.random.random() > self.GAMMA:
            print('AWESOMEEEEEEEEEEEEEEEEEEEEEE')
            self.GAMMA -= ENTROPY_BETA
            return actions.FunctionCall(self.last_action[0], self.last_action[1])
        else:
            action = np.random.choice(obs.observation.available_actions)
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[action].args]
            self.last_action += [action, args]
            return actions.FunctionCall(action, args)
