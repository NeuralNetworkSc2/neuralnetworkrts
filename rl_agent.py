import random
from pysc2.lib.colors import categorical
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, float64, func, nn, return_types
from pysc2.env.environment import StepType, TimeStep
import pysc2
from pysc2.agents import base_agent
from pysc2.lib import actions, features
import numpy as np
import collections

SCREEN_SIZE = 64

GAMMA = 0.01
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


class CategoryEmbedder(nn.Module):
    def __init__(self, categories):
        super().__init__()

        self.category_embeddings = []
        one_dim = 0
        high_dim = 0
        for i, cat in enumerate(categories):
            if cat.scale == 2:
                em = None
                one_dim += 1
            else:
                high_dim += 1
                em = nn.Embedding(cat.scale, 16)
                self.add_module(f"cat {i}", em)

            self.category_embeddings.append(em)

        self.output_shape = one_dim + 16
        self.relu = nn.ELU()

    def forward(self, inputs):
        float_inputs = inputs.float()
        unbound = inputs.unbind(
            dim=1)
        f_unbound = float_inputs.unbind(dim=1)
        result = []
        two_value = []
        for u, f, em in zip(unbound, f_unbound, self.category_embeddings):
            if em is None:
                two_value.append(f.unsqueeze(1))
            else:
                result.append(
                    em(u).permute(0, 3, 1, 2).unsqueeze(0)
                )

        reduced = torch.cat(result, dim=0).sum(dim=0)
        extended = torch.cat([reduced] + two_value, dim=1)

        return extended


class ResidualFiLMBlock(nn.Module):
    def __init__(self, in_layers, size, gating_size):
        super().__init__()
        self.c1 = nn.Conv2d(in_layers, in_layers, (3, 3), padding=1)
        self.n1 = nn.LayerNorm((in_layers, size, size))
        self.c2 = nn.Conv2d(in_layers, in_layers, (3, 3), padding=1)
        self.n2 = nn.LayerNorm((in_layers, size, size))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.film = FiLM((in_layers, size, size), gating_size)

    def forward(self, x, gating):
        old_x = x
        x = self.n1(x)
        x = self.c1(x)
        x = self.relu1(x)
        x = self.c2(x)
        x = self.n2(x)
        x = self.film(x, gating)
        x = self.relu2(x)

        return x + old_x


class ActionOut(nn.Module):
    def __init__(self, input_size) -> None:
        super(ActionOut, self).__init__()
        self.inp = nn.Linear(input_size, 256)
        self.out = nn.GLU()

    def forward(self, x, gating):
        x = self.inp(x)

        return self.out(x + gating)


class ControlGrOut(nn.Module):
    def __init__(self) -> None:
        super(ControlGrOut, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1024, 256), nn.ELU())
        self.layer2 = nn.Linear(256, 10)
        self.encoder = nn.Linear(21, 10)
        self.position_encoder = nn.Parameter(torch.randn(1, 1, 10, 4))
        self.action = nn.Linear(21 + 256, 5)

    def forward(self, inp, control_groups):
        inp = self.layer1(inp)
        select = self.layer2(inp)
        values = self.encoder(control_groups)
        values = torch.cat(
            [values, self.position_encoder.repeat(values.shape[:2] + (1, 1))], dim=-1)
        print(values.shape)
        attention = values @ select.unsqueeze(-1)
        selected = nn.functional.softmax(
            attention, dim=-2).detach() * control_groups
        act_in = torch.cat([inp, selected], dim=-1)
        return attention, self.action(act_in)


class ScreenInput(nn.Module):
    def __init__(self, scalar_input, categorical_input):
        self.input_column = []

        self.cat_embedder = CategoryEmbedder(categorical_input)
        scales = []
        for x in scalar_input:
            scales.append(x.scale - 1)

        self.scales = torch.tensor(scales).unsqueeze(0).unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1).cuda()

        self.reshaper = nn.Sequential(
            nn.Conv2d(len(scalar_input) + self.cat_embedder.output_shape,
                      32, (1, 1),
                      padding=0), nn.ELU())

        self.reduce1 = nn.Sequential(
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ELU())  # 32x32
        self.reduce2 = nn.Sequential(
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1), nn.ELU())  # 16x16
        self.reduce3 = nn.Sequential(
            nn.Conv2d(128, 128, (4, 4), stride=2, padding=1), nn.ELU())  # 8x8

        filters = 128
        for i in range(2):
            self.input_column.append(
                ResidualFiLMBlock(filters, 8, 512))

        self.layer_norm_out = nn.LayerNorm((128, 8, 8))

        for i, mod in enumerate(self.input_column):
            self.add_module(f"layer {i}", mod)

    def forward(self, scalar, categorical, gating):

        shape = scalar.shape
        shape2 = categorical.shape
        scalar = scalar / self.scales
        scalar = scalar.view((shape[0] * shape[1], ) + shape[2:])
        categorical = categorical.view((shape2[0] * shape2[1], ) + shape2[2:])
        categorical = self.cat_embedder(categorical)
        bypass = []

        x = torch.cat([scalar, categorical], dim=1)
        x = self.reshaper(x)
        bypass.append(x)
        x = self.reduce1(x)
        bypass.append(x)
        x = self.reduce2(x)
        bypass.append(x)
        x = self.reduce3(x)
        gating = gating.view(-1, 512)
        i = 0
        for layer in self.input_column:
            x = layer(x, gating)
            i += 1

        return x.view(shape[:2] + (128, 8, 8)), bypass


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
        self.lstm = nn.LSTM(512 + 512 + 512, 512,
                            2)

        categorical = []
        scalar = []
        for feature in features.SCREEN_FEATURES:
            if feature.type == features.FeatureType.CATEGORICAL:
                categorical.append(feature)
            else:
                scalar.append(feature)

        minimap_cat = []
        minimap_sc = []
        for feature in features.SCREEN_FEATURES:
            if feature.type == features.FeatureType.CATEGORICAL:
                minimap_cat.append(feature)
            else:
                minimap_sc.append(feature)
        in_type = {"screen": (scalar, categorical),
                   "minimap": (minimap_sc, minimap_cat)}

        scalar, categorical = in_type["screen"]
        self.screen_in = ScreenInput(scalar, categorical)
        scalar, categorical = in_type["screen"]
        self.map_in = ScreenInput(scalar, categorical)
        self.output = {}
        self.embedding = {}
        self.output["function"] = ActionOut(4 * 512)
        self.embedding["function"] = nn.Embedding(len(actions.FUNCTIONS), 1024)
        self.output["control_groups"] = ControlGrOut()

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


class A3CAgent(base_agent.BaseAgent):
    def __init__(self, net: Model):
        super(A3CAgent, self).__init__()
        self.net = net
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=0.26, eps=1e-3)
        self.exp_source = []
        self.last_action = [0, []]
        self.GAMMA = 0.99

    def setup(self, obs_spec, action_spec):
        super(A3CAgent, self).setup(obs_spec, action_spec)

    def can_do(self, obs, action):
        return action in obs.observation['available_actions']

    def discount_rewards(self, reward, dones):
        discounted = np.zeros_like(reward)
        running_add = 0
        for i in reversed(range(0, len(reward))):
            running_add = running_add * GAMMA * (1 - dones[i]) + reward[i]
            discounted[i] = running_add

        if np.std(discounted) != 0:
            discounted -= np.mean(discounted)  # normalizing the result
            discounted /= np.std(discounted)  # divide by standard deviation

        return discounted

    def replay(self,):
        pass

    def step(self, obs: TimeStep):
        super(A3CAgent, self).step(obs)
        # print('last action: ', obs.observation['last_actions'])
        # if len(obs.observation['last_actions']) == 0 and len(self.last_action) != 0:
        #     self.last_action.pop()
        # print('after pop', self.last_action[-1])
        self.exp_source.append(Experience(
            obs.step_type, obs.reward, self.last_action, torch.FloatTensor(obs.observation['feature_screen'])))
        states_v, actions_t, vals_ref_v = unpack_batch(
            self.exp_source, self.net)
        self.exp_source.clear()

        self.optimizer.zero_grad()
        logits_v, value_v = self.net(states_v)
        print(value_v)

        loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

        log_prob_v = F.log_softmax(logits_v, dim=1)
        print(log_prob_v)
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
        # print(actions_t)
        action = random.choice(actions_t[0])
        # print('action result: ', obs.observation['last_actions'])
        if np.random.random() > self.GAMMA:
            # print('AWESOMEEEEEEEEEEEEEEEEEEEEEE')
            # print(action)
            if self.GAMMA > 0.05:
                self.GAMMA -= ENTROPY_BETA
            print(type(action).__name__)
            if type(action).__name__ == 'int32' or type(action).__name__ == 'int':
                if self.can_do(obs, action):
                    # print('action is int')
                    args = [[np.random.randint(0, size) for size in arg.sizes]
                            for arg in self.action_spec.functions[action].args]
                    return actions.FunctionCall(action, args)
            elif (len(action) == 2):
                # print('len = 2')
                if self.can_do(obs, action[0]):
                    return actions.FunctionCall(action[0], action[1])
            elif (len(action) == 1):
                # print('len = 1')
                args = [[np.random.randint(0, size) for size in arg.sizes]
                        for arg in self.action_spec.functions[action[0][0]].args]
                if self.can_do(obs, action[0]):
                    return actions.FunctionCall(action[0], args)
            else:
                return actions.FunctionCall(0, [])
        else:
            action = np.random.choice(obs.observation.available_actions)
            # print('action choose', action)
            args = [[np.random.randint(0, size) for size in arg.sizes]
                    for arg in self.action_spec.functions[action].args]
            # print('args for action', args)
            # if self.last_action.__len__() < 100:
            self.last_action.append([action, args])
            # print(self.last_action[-1])
            return actions.FunctionCall(action, args)
        return actions.FunctionCall(0, [])
