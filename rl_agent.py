import random
from pysc2 import lib
from pysc2.lib.colors import categorical
import torch
import torch.nn as nn
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor, cummax, float64, nn
from pysc2.env.environment import StepType, TimeStep
import pysc2
from pysc2.agents import base_agent
from pysc2.lib import actions, features, static_data
import numpy as np
import collections
from pysc2.lib.features import ScoreCumulative as sc

from pickle import load, loads
from zstd import decompress


GAMMA = 0.01
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 1
NUM_ENVS = 50

REWARD_STEPS = 4
CLIP_GRAD = 0.1


def weighted_score(score, reward):
    return np.array([
        score[sc.killed_value_units] * 0.3,
        score[sc.killed_value_structures]*0.8,
        score[sc.collected_minerals] * 0.1,
        score[sc.collected_vespene] * 0.5,
        reward * 1000.0
    ])


class GLU(nn.Module):
    def __init__(self, input_size, gating_size, output_size):
        super(GLU, self).__init__()
        self.gate = nn.Linear(gating_size, input_size)
        self.lin = nn.Linear(input_size, output_size)

    def forward(self, x, gating):
        g = torch.sigmoid(self.gate(gating))
        return self.lin(g * x)


class Experience(collections.namedtuple(
        'Experience', ['step_type', 'reward', 'action', 'observation', 'map'])):

    def first(self):
        return self.step_type is StepType.FIRST

    def mid(self):
        return self.step_type is StepType.MID

    def last(self):
        return self.step_type is StepType.LAST


class Core(nn.Module):
    def __init__(self) -> None:
        super(Core, self).__init__()
        self.conv_screen = nn.Sequential(
            nn.Conv2d(128, 32, (4, 4), stride=2,
                      padding=1), nn.ReLU(), nn.LayerNorm((32, 4, 4))
        )
        self.conv_map = nn.Sequential(
            nn.Conv2d(128, 32, (4, 4), stride=2,
                      padding=1), nn.ReLU(), nn.LayerNorm((32, 4, 4))

        )
        self.norm = nn.LayerNorm(128)
        self.transform = nn.TransformerEncoderLayer(
            128, 4, 512, dropout=0.0)
        self.lstm = nn.LSTM(128, 128, 1)

    def forward(self, screen, map, state):
        device = torch.device("cpu")
        shape = screen.shape
        screen = screen.transpose(2, 4).reshape(
            shape[0] * shape[1], -1, shape[2]).to(device)
        map = map.transpose(2, 4).reshape(
            shape[0] * shape[1], -1, shape[2]).to(device)

        input = torch.cat([screen, map], dim=1).to(device)
        input = self.norm(input)
        input = input.transpose(0, 1)
        input = self.transform(input)
        input = input.transpose(0, 1).to(device)
        new_state = []
        for items in state:
            items = items.to("cpu")
            new_state.append(items)

        lstm_in = input.reshape(
            shape[0], shape[1] * shape[3] * shape[4] * 2, shape[2]).to(device)
        input, next_state = self.lstm(lstm_in, new_state)

        input = input + lstm_in

        input = self.norm(input)
        input = input.transpose(0, 1)
        input = input.reshape(shape[0] * shape[1], 2,
                              shape[3], shape[4], shape[2]).transpose(2, 4)
        screenout = input[:, 0]
        mapout = input[:, 1]
        screenout_reduced = self.conv_screen(
            screenout).view(shape[0], shape[1], -1)
        mapout_reduced = self.conv_map(mapout).view(shape[0], shape[1], -1)
        return screenout.view(shape), mapout.view(shape), torch.cat([screenout_reduced, mapout_reduced], dim=2), next_state


class FiLM(nn.Module):
    def __init__(self, output_size, gating_size):
        super(FiLM, self).__init__()
        self.scale = nn.Linear(gating_size, output_size[0])
        self.shift = nn.Linear(gating_size, output_size[0])

    def forward(self, x, gating):
        scale = self.scale(gating).unsqueeze(-1).unsqueeze(-1)
        shift = self.shift(gating).unsqueeze(-1).unsqueeze(-1)
        return scale * x + shift


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


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_layers):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(in_layers, in_layers), nn.ELU())
        self.n1 = nn.LayerNorm((in_layers, ))
        self.l2 = nn.Sequential(nn.Linear(in_layers, in_layers), nn.ELU())
        self.n2 = nn.LayerNorm((in_layers, ))

    def forward(self, x):
        old_x = x
        x = self.n1(x)
        x = self.l1(x)
        x = self.n2(x)
        x = self.l2(x)

        return x + old_x


class ActionOut(nn.Module):
    def __init__(self, input_size, count, gating_size) -> None:
        super(ActionOut, self).__init__()
        self.inp = nn.Linear(input_size, 256)

        self.layers = []

        for i in range(4):
            self.layers.append(ResidualDenseBlock(256))
            self.add_module(f"layer {i}", self.layers[i])

        self.out = GLU(256, gating_size, count)

    def forward(self, x, gating):
        x = self.inp(x)
        for layer in self.layers:
            x = layer(x)

        return self.out(x, gating)


class ControlGroupsInput(nn.Module):
    def __init__(self):
        super(ControlGroupsInput, self).__init__()
        self.layer_size = 21
        self.unit_embedder = nn.Embedding(
            max(static_data.UNIT_TYPES) + 1, 10)
        self.layer_norm = nn.LayerNorm((64, ))
        self.layer = nn.Sequential(
            nn.Linear(self.layer_size * 10, 64), nn.ReLU())

    def forward(self, x, hint, norm=True):
        x = torch.tensor(x)
        embedded = self.unit_embedder(torch.tensor(x[:, :, :, 0]).long())
        embedded_hint = self.unit_embedder(
            torch.tensor(hint[:, :, :, 0]).long())
        embedded_hint = embedded_hint.repeat(
            embedded.shape[0] // embedded_hint.shape[0], 1, 1, 1)
        x = torch.cat([embedded, x[:, :, :, 1:].float(), embedded_hint],
                      dim=-1)
        bypass = x
        s = x.shape
        x = self.layer(x.view(s[:2] + (10 * self.layer_size, )))
        if norm:
            x = self.layer_norm(x)
        return x, bypass


class ControlGrOut(nn.Module):
    def __init__(self) -> None:
        super(ControlGrOut, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(1024, 256), nn.ELU())
        self.layer2 = nn.Linear(256, 14)
        self.encoder = nn.Linear(21, 10)
        self.position_encoder = nn.Parameter(torch.randn(1, 1, 10, 4))
        self.action = nn.Linear(21 + 256, 5)

    def forward(self, inp, control_groups):
        inp = self.layer1(inp)
        select = self.layer2(inp)
        values = self.encoder(control_groups)
        values = torch.cat(
            [values, self.position_encoder.repeat(values.shape[:2] + (1, 1))], dim=-1)
        attention = values @ select.unsqueeze(-1)
        selected = nn.functional.softmax(
            attention, dim=-2).detach() * control_groups
        act_in = torch.cat([inp, selected], dim=-1)
        return attention, self.action(act_in)


class SimpleInput(nn.Module):
    def __init__(self, in_shape):
        super().__init__()
        self.cut = len(in_shape)
        self.size = 1
        self.layer_norm = nn.LayerNorm((64, ))
        for x in in_shape:
            self.size *= x
        self.layer = nn.Sequential(nn.Linear(self.size, 64),
                                   nn.ReLU())
        self.layer2 = nn.Sequential(nn.Linear(64, 64),
                                    nn.ReLU())

    def forward(self, x, norm=True):
        s = x.shape[:-self.cut] + (self.size, )
        x = self.layer(x.view(s).float())
        x = self.layer2(x)
        if norm:
            x = self.layer_norm(x)
        return x


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
                    em(u.long()).permute(0, 3, 1, 2).unsqueeze(0)
                )

        reduced = torch.cat(result, dim=0).sum(dim=0)
        extended = torch.cat([reduced] + two_value, dim=1)

        return extended


class ScreenInput(nn.Module):
    def __init__(self, scalar_input, categorical_input):
        super(ScreenInput, self).__init__()
        self.input_column = []
        self.cat_embedder = CategoryEmbedder(categorical_input)
        scales = []
        for x in scalar_input:
            scales.append(x.scale - 1)

        self.scales = torch.tensor(scales).unsqueeze(0).unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1)

        self.reshaper = nn.Sequential(
            nn.Conv2d(len(scalar_input),
                      32, (1, 1),
                      padding=0), nn.ELU())

        self.reduce1 = nn.Sequential(
            nn.Conv2d(32, 64, (4, 4), stride=2, padding=1),
            nn.ELU())
        self.reduce2 = nn.Sequential(
            nn.Conv2d(64, 128, (4, 4), stride=2, padding=1), nn.ELU())
        self.reduce3 = nn.Sequential(
            nn.Conv2d(128, 128, (4, 4), stride=2, padding=1), nn.ELU())

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
        scalar = scalar / self.scales.cpu()
        scalar = scalar.view((shape[0] * shape[1], ) + shape[2:])
        bypass = []
        categorical = categorical.view((shape2[0] * shape2[1], ) + shape2[2:])
        categorical = self.cat_embedder(categorical)

        x = torch.cat([scalar, categorical], dim=1)
        x = self.reshaper(scalar)
        bypass.append(x)
        x = self.reduce1(x)
        bypass.append(x)
        x = self.reduce2(x)
        bypass.append(x)
        x = self.reduce3(x)
        gating = gating.view(-1, 512)
        for layer in self.input_column:
            x = layer(x, gating)

        return x.view(shape[:2] + (128, 8, 8)), bypass


Feature = collections.namedtuple("Feature", ["id", "shape"
                                             ])


def get_categorical_scalar(feature):
    categorical = []
    scalar = []

    for feat in feature:
        if feat.type == features.FeatureType.CATEGORICAL:
            categorical.append(Feature(feat.index, feat.scale))
        else:
            scalar.append(Feature(feat.index, feat.scale))

    return categorical, scalar


screen_categorical, screen_scalar = get_categorical_scalar(
    features.SCREEN_FEATURES)
minimap_categorical, minimap_scalar = get_categorical_scalar(
    features.MINIMAP_FEATURES)


def get_features_by_type(input, feat_list, normalize=False):
    output = []
    for feat in feat_list:
        # print(feat)
        if normalize:
            norm = feat.shape
        else:
            norm = 1
        output.append(input[feat.id] / norm)
    return np.asarray(output)


class ListInput(nn.Module):
    def __init__(self, in_shape, embedder_size):
        super(ListInput, self).__init__()
        self.encoder = nn.Linear(in_shape[1] + 10 - 1,
                                 64)
        self.unit_embedder = nn.Embedding(embedder_size, 10)
        self.layer_norm = nn.LayerNorm((64, ))
        self.modifier = nn.Sequential(
            nn.Linear(64, 64), nn.ELU())

    def forward(self, x, norm=True):
        x = torch.Tensor(x)
        embedded = self.unit_embedder(x[:, 0].long())
        x = torch.cat([embedded, x[1:].float()], dim=0)
        x = self.encoder(x)
        v = x.max(dim=-2, keepdim=False).values

        x = self.modifier(v)
        if norm:
            x = self.layer_norm(x)
        return x


class ScreenOutput(nn.Module):
    def __init__(self):
        super(ScreenOutput, self).__init__()
        self.output_column = []
        self.deconv_layers = []

        self.reencode = nn.Conv2d(128 + 16, 128, kernel_size=(1, 1))

        for i in range(3):
            self.output_column.append(ResidualFiLMBlock(128, 8, 1024))
            self.add_module(f"layer {i}", self.output_column[i])

        self.deconv_layer1 = nn.Sequential(
            nn.ConvTranspose2d(128,
                               128,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=1), nn.ELU())
        self.deconv_layer2 = nn.Sequential(
            nn.LayerNorm((128, 16, 16)),
            nn.ConvTranspose2d(128,
                               64,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=1), nn.ELU())

        self.deconv_layer3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=(4, 4), stride=2,
                               padding=1))
        self.output_layer = nn.Sequential(
            nn.ELU(), nn.Conv2d(32, 1, kernel_size=(1, 1), stride=1,
                                padding=0))

    def forward(self, spatial_in, lstm_in):
        spatial_in, bypass = spatial_in
        shape = spatial_in.shape
        lstm_in_shredded = lstm_in.view((-1, 16, 8, 8))

        spatial_in = spatial_in.view((shape[0] * shape[1], ) + shape[2:])

        shape2 = lstm_in.shape
        lstm_in = lstm_in.view((shape2[0] * shape2[1], ) + shape2[2:])
        print(spatial_in.shape, lstm_in_shredded.shape)
        x = torch.cat((spatial_in, lstm_in_shredded), dim=1)
        x = self.reencode(x)

        i = 0
        for layer in self.output_column:
            x = layer(x, lstm_in)
            i += 1

        x = self.deconv_layer1(x)
        x += bypass[2]
        x = self.deconv_layer2(x)
        x += bypass[1]
        x = self.deconv_layer3(x)
        x += bypass[0]
        x = self.output_layer(x)

        return x.view(shape[:-3] + (64 * 64, ))


_TRACKED_FEATURES = \
    {'build_queue': (0, 7),
     'cargo': (0, 7),
     'control_groups': (10, 2),
     'multi_select': (0, 7),
     'player': (11,),
     'production_queue': (0, 2),
     'single_select': (0, 7),
     'feature_screen': (27, 64, 64),
     'feature_minimap': (11, 64, 64),
     'game_loop': (1,),
     'available_actions': (573,)}

TARGET_COUNT = 32


def insert_tcount(t):
    out = []
    for x in t:
        if x == 0:
            out.append(TARGET_COUNT)
        else:
            out.append(x)
    return tuple(out)


TRACKED_FEATURES = {
    x: insert_tcount(_TRACKED_FEATURES[x])
    for x in _TRACKED_FEATURES
}


def get_max_val(values):
    maxim = 0
    for val in values:
        maxim = max(maxim, val.shape)
    return maxim


def get_type(feats):
    x = get_max_val(feats)
    if x < 256:
        return np.uint8
    else:
        return np.int16


t_screen_cat = get_type(screen_categorical)
t_screen_sca = get_type(screen_scalar)
t_minimap_cat = get_type(minimap_categorical)
t_minimap_sca = get_type(minimap_scalar)


def get_observation(obs):
    out_dict = {}
    for x in TRACKED_FEATURES:
        shape = TRACKED_FEATURES[x]
        if "screen" in x:
            screen = obs[x]
            sc = get_features_by_type(screen, screen_scalar)
            cat = get_features_by_type(screen, screen_categorical)

            sc = sc.reshape((1, 1) + sc.shape)
            cat = cat.reshape((1, 1) + cat.shape)

            out_dict[x + "_scalar"] = sc.astype(t_screen_sca)
            out_dict[x + "_categorical"] = cat.astype(t_screen_cat)
        elif "minimap" in x:
            minimap = obs[x]
            sc = get_features_by_type(minimap, minimap_scalar)
            cat = get_features_by_type(minimap, minimap_categorical)

            sc = sc.reshape((1, 1) + sc.shape)
            cat = cat.reshape((1, 1) + cat.shape)

            out_dict[x + "_scalar"] = sc.astype(t_minimap_sca)
            out_dict[x + "_categorical"] = cat.astype(t_minimap_cat)
        elif "available_actions" in x:
            thing = np.zeros((573, ), dtype=np.float32)
            for data in obs[x]:
                thing[data] = 1

            out_dict[x] = thing.reshape((1, 1, 573))
        elif "game_loop" in x:
            thing = np.zeros((1, ), dtype=np.float32)
            thing[0] = np.minimum(obs[x] / (22.4 * 60 * 20), 10)
            out_dict[x] = thing.reshape((1, 1, 1))
        else:
            out_dict[x] = obs[x].copy()
            if "player" in x:
                out_dict[x] = np.log(out_dict[x] + 1)

            out_dict[x].resize((1, 1) + shape)

    return out_dict


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(512 * 3, 1024, 2)
        self.lstm_reshape_gating = GLU(2048, 64 * 4, 1024)
        self.core = Core()
        categorical = []
        scalar = []
        for feature in features.SCREEN_FEATURES:
            if feature.type == features.FeatureType.CATEGORICAL:
                categorical.append(feature)
            else:
                scalar.append(feature)

        minimap_cat = []
        minimap_sc = []
        self.input_columns = {}
        for feature in features.MINIMAP_FEATURES:
            if feature.type == features.FeatureType.CATEGORICAL:
                minimap_cat.append(feature)
            else:
                minimap_sc.append(feature)
        in_type = {"screen": (scalar, categorical),
                   "minimap": (minimap_sc, minimap_cat)}

        scalar_input_size = 0
        for name in TRACKED_FEATURES:
            shape = TRACKED_FEATURES[name]
            if "screen" in name or "minimap" in name:
                continue
            elif "build_order" in name:
                self.input_columns[name] = nn.Sequential(
                    nn.Embedding(max(static_data.UNIT_TYPES) + 1, 10),
                    SimpleInput((20, 10)))
                scalar_input_size += 64
            elif "control_group" in name:
                self.input_columns[name] = ControlGroupsInput()
                scalar_input_size += 64

            elif "prev_action" in name:
                self.input_columns[name] = nn.Sequential(
                    nn.Embedding(573, 16), SimpleInput((16, )))
                scalar_input_size += 64
            elif len(shape) == 2 and shape[-2] == 0:

                if shape[-1] == 2:
                    embedder_len = max(actions.ABILITY_IDS) + 1
                else:
                    embedder_len = max(static_data.UNIT_TYPES) + 1

                self.input_columns[name] = ListInput(shape, embedder_len)
                scalar_input_size += 64
            else:
                self.input_columns[name] = SimpleInput(shape)
                scalar_input_size += 64

            self.add_module(name + " column", self.input_columns[name])

        self.scalar_reshape = nn.Linear(scalar_input_size, 512)
        scalar, categorical = in_type["screen"]
        self.screen_in = ScreenInput(scalar, categorical)
        scalar, categorical = in_type["minimap"]
        self.map_in = ScreenInput(scalar, categorical)
        self.embedding = {}
        self.output_column = {}
        self.output_column["function"] = ActionOut(
            inputs_size=2560, count=len(actions.FUNCTIONS), gating_size=256)
        self.embedding["function"] = nn.Embedding(len(actions.FUNCTIONS), 1024)
        self.control_groups_out = ControlGrOut()
        for x in actions.TYPES:
            u = str(x)
            if "queued" in u:
                self.embedding[u] = nn.Embedding(x.sizes[0], 1024)
                self.add_module(u + " embedding", self.embedding[u])

            if "screen" in u or "minimap" in u:
                self.output_column[u] = ScreenOutput()
                self.add_module(u + " out", self.output_column[u])
        self.add_module("function embedding", self.output_column["function"])
        self.add_module("control_groups embedding", self.control_groups_out)

    def forward(self, inputs, hidden, targets):
        sample = next(iter(inputs.values()))
        gating_input = torch.zeros(sample.shape[:2] + (0, ))
        scalar_input = torch.zeros(sample.shape[:2] + (0, ))
        for name in sorted(TRACKED_FEATURES):
            if "screen" in name or "minimap" in name:
                continue
            elif "control_group" in name:
                x, control_groups = self.input_columns[name](
                    inputs[name], torch.zeros(inputs[name].shape))
                scalar_input = torch.cat((scalar_input, x), 2)

            elif ("available_actions" in name or "build_order" in name
                  or "select" in name):
                x = self.input_columns[name](inputs[name])
                scalar_input = torch.cat((scalar_input, x), 2)
                gating_input = torch.cat((gating_input, x), 2)
            else:
                x = self.input_columns[name](inputs[name])
                scalar_input = torch.cat((scalar_input, x), 2)
        scalar_input = self.scalar_reshape(scalar_input)
        screen_inp, screen_bypass = self.screen_in(
            inputs["feature_screen_scalar"], inputs["feature_screen_categorical"], scalar_input)
        hidden_1 = hidden[:2]
        hidden_2 = hidden[2:]

        minimap_inp, minimap_bypass = self.map_in(
            inputs["feature_minimap_scalar"], inputs["feature_minimap_categorical"], scalar_input)

        screen_out, map_out, lstm_input, next_hidden = self.core(
            screen_inp, minimap_inp, hidden_1)

        lstm_input = torch.cat([lstm_input, scalar_input], dim=2)
        lstm_input = lstm_input.cpu()
        hidden_2 = [item.cpu() for item in hidden_2]
        print(lstm_input.shape)
        lstm_output, next_hidden2 = self.lstm(lstm_input, hidden_2)
        lstm_output = torch.cat([lstm_output, lstm_input], dim=2)

        screen = (screen_out, screen_bypass)
        minimap = (map_out, minimap_bypass)

        output, lstm_out = self.chain(
            lstm_output, scalar_input, targets)

        for x in self.output_column:
            if x in ["function", "time_skip", str(actions.TYPES.queued)] or x == "value":
                continue
            if "screen" in x:
                output[x] = self.output_column[x](
                    screen, lstm_output)
            elif "minimap" in x:
                output[x] = self.output_column[x](
                    minimap, lstm_output)
            elif "control_group" in x:
                output[self.control_group_id_name], output[
                    self.control_group_act_name] = self.output_column[x](
                        lstm_output, control_groups)
            else:
                output[x] = self.output_column[x](lstm_output)

        keys = list(output.keys())
        if not self.training:
            for x in keys:
                if x in ["function", "time_skip", str(actions.TYPES.queued)] or "_sampled" in x:
                    continue
                output[x + "_sampled"] = self.sample_action(
                    output[x], x)
        return output, next_hidden + next_hidden2

    def sample_action(self, ps, name):
        ps = torch.softmax(ps, dim=-1)
        indices = torch.distributions.Categorical(ps.squeeze(0)).sample()
        action = indices.unsqueeze(0)
        return action

    def chain(self, lstm_output, gating, targets):
        outputs = {}
        key = "function"
        gating = gating.reshape(-1, 256)
        result = self.output_column[key](lstm_output, gating)
        lstm_output = self.lstm_reshape_gating(lstm_output, gating)

        if "value" in self.output_column:
            outputs["value"] = self.output_column["value"](lstm_output)

        action = targets[key]
        print(action)

        lstm_output = lstm_output + self.embedding[key](action.long())
        outputs[key] = result
        outputs[key + "_sampled"] = action

        for key in [str(actions.TYPES.queued)]:
            # result = self.output_column["queued"](lstm_output)

            action = targets[key]

            lstm_output = lstm_output + self.embedding[key](action.long())

            # outputs[key] = result
            outputs[key + "_sampled"] = action

        return outputs, lstm_output


def unpack_batch(obs, batch, net, device='cpu'):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    map = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(np.array(exp.observation, copy=False))
        actions.append(exp.action)
        rewards.append(exp.reward)
        map.append(np.array(exp.map))
        if not exp.step_type == StepType.LAST:
            not_done_idx.append(idx)
            last_states.append(np.array(exp.step_type, copy=False))
    states_v = torch.FloatTensor(np.array(states, copy=False)).to(device)
    map_v = torch.FloatTensor(np.array(map, copy=False)).to(device)
    # actions_t = torch.LongTensor(actions).to(device)
    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_vals_v = net(obs, states_v, map_v)[1]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, -1]
        rewards_np[not_done_idx] += GAMMA ** REWARD_STEPS * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions, ref_vals_v, map_v


class A3CAgent(base_agent.BaseAgent):
    def __init__(self, net: Model):
        super(A3CAgent, self).__init__()
        self.net = net
        self.optimizer = optim.Adam(
            self.net.parameters(), lr=0.26, eps=1e-3)
        self.exp_source = []
        self.last_action = [0, []]
        self.GAMMA = 0.99
        self.step_count = 0
        self.replays = []
        with open(r"/home/gilsson/replay_save/1", "rb") as file:
            self.replays.append(loads(decompress(load(file))))

        inputs, targets, masks, hidden = zip(*self.replays)
        self.current_replay = (
            inputs[:100], targets, masks[:100], hidden)

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

        print('Abobaaaaaaaaaaaaaaaaaa')

        def concat(x):
            output = {}
            for entry in x[0]:
                output[entry] = torch.cat([p[entry] for p in x], axis=1)

            return output

        def concat_lstm_hidden(x):
            result = tuple()
            swapped = zip(*x)
            for field in swapped:
                print(field, type(field))
                output = torch.cat(field, axis=1)
                result = result + (output, )

            return result

        inputs = concat(self.current_replay[0])
        hiddens = concat_lstm_hidden(self.current_replay[3])
        targets = concat(self.current_replay[2])
        masks = concat(self.current_replay[1])

        out = self.net(inputs, hiddens, targets)
        step_type = obs.step_type
        reward = obs.reward
        obs_new = get_observation(obs.observation)
        # print(obs.observation['feature_screen'][-1].shape)
        # print('last action: ', obs.observation['last_actions'])
        # if len(obs.observation['last_actions']) == 0 and len(self.last_action) != 0:
        #     self.last_action.pop()
        # print('after pop', self.last_action[-1])
        # self.exp_source.append(Experience(
        #     step_type, reward, self.last_action, torch.FloatTensor(obs.observation['feature_screen']), torch.FloatTensor(obs.observation['feature_minimap'])))
        # states_v, actions_t, vals_ref_v, map_v = unpack_batch(obs,
        #                                                       self.exp_source, self.net)
        # self.exp_source.clear()

        # self.optimizer.zero_grad()
        # logits_v, value_v = self.net(obs, states_v, map_v)
        # print(value_v)

        # loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
        #
        # log_prob_v = F.log_softmax(logits_v, dim=1)
        # # print(log_prob_v)
        # adv_v = vals_ref_v - value_v.squeeze(-1).detach()
        # log_prob_actions_v = adv_v * \
        #     log_prob_v[range(BATCH_SIZE), actions_t[0][0]]
        # loss_policy_v = -log_prob_actions_v.mean()
        #
        # prob_v = F.softmax(logits_v, dim=1)
        # entropy_loss_v = ENTROPY_BETA * \
        #     (prob_v * log_prob_v).sum(dim=1).mean()
        #
        # # calculate policy gradients only
        # loss_policy_v.backward(retain_graph=True)
        # grads = np.concatenate([p.grad.data.cpu().numpy().flatten()
        #                         for p in self.net.parameters()
        #                         if p.grad is not None])
        #
        # # apply entropy and value gradients
        # loss_v = entropy_loss_v + loss_value_v
        # loss_v.backward()
        # self.optimizer.step()
        # # get full loss
        # loss_v += loss_policy_v
        # # print(actions_t)
        # action = random.choice(actions_t[0])
        # print('action result: ', obs.observation['last_actions'])
        # if np.random.random() > self.GAMMA:
        #     # print('AWESOMEEEEEEEEEEEEEEEEEEEEEE')
        #     # print(action)
        #     if self.GAMMA > 0.05:
        #         self.GAMMA -= ENTROPY_BETA
        #     print(type(action).__name__)
        #     if type(action).__name__ == 'int32' or type(action).__name__ == 'int':
        #         if self.can_do(obs, action):
        #             # print('action is int')
        #             args = [[np.random.randint(0, size) for size in arg.sizes]
        #                     for arg in self.action_spec.functions[action].args]
        #             return actions.FunctionCall(action, args)
        #     elif (len(action) == 2):
        #         # print('len = 2')
        #         if self.can_do(obs, action[0]):
        #             return actions.FunctionCall(action[0], action[1])
        #     elif (len(action) == 1):
        #         # print('len = 1')
        #         args = [[np.random.randint(0, size) for size in arg.sizes]
        #                 for arg in self.action_spec.functions[action[0][0]].args]
        #         if self.can_do(obs, action[0]):
        #             return actions.FunctionCall(action[0], args)
        #     else:
        #         return actions.FunctionCall(0, [])
        # else:
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
