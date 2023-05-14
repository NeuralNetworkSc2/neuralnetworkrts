import pickle
from rl_agent import get_observation
from pysc2.lib import features
from pysc2.lib import units
from pysc2 import run_configs
from pysc2.env import sc2_env
from zstd import compress
from pysc2.run_configs.lib import RunConfig
from s2clientprotocol import sc2api_pb2 as sc_pb
from torch import set_num_threads
import os
from torch.multiprocessing import Process
from pickle import dump
import torch
from zstd import compress
from os import path
from main import AGENT_INTERFACE_FORMAT
from absl import flags
import numpy as np
from pysc2.lib import actions as ac

screen_size = (64, 64)
minimap_size = (64, 64)
NUM_WORKERS = 2


def generate_zeroed_dicts():
    mask_dict = {}
    result_dict = {}

    result_dict["function"] = np.zeros((1, 1), dtype=np.int32)
    mask_dict["function"] = np.zeros((1, 1))
    result_dict["time_skip"] = np.zeros((1, 1), dtype=np.int32)
    mask_dict["time_skip"] = np.zeros((1, 1))

    for t in ac.TYPES:
        x = str(t)
        result_dict[x] = np.zeros((1, 1), dtype=np.int32)
        mask_dict[x] = np.zeros((1, 1))

    return result_dict, mask_dict


def extract_action(fun_call):
    if fun_call is None:
        fun_id = 0
    else:
        fun_id = fun_call.function

    rd, md = generate_zeroed_dicts()

    rd["function"][0, 0] = fun_id
    md["function"][0, 0] = 1.0
    md["time_skip"][0, 0] = 1.0
    rd["time_skip"][0, 0] = 0
    if fun_id == 0:
        return rd, md

    for data, argt in zip(
            fun_call.arguments,
            ac.FUNCTION_TYPES[ac.FUNCTIONS[int(fun_id)].function_type]):
        a_id = data[0]
        shape = argt.sizes
        if len(shape) > 1:
            a_id = data[0] * 64 + data[1]

        name = str(argt)
        rd[name][0, 0] = a_id
        md[name][0, 0] = 1.0

    return rd, md


def to_tensor(x, type=torch.float32):
    result = {}
    for field in x:
        if "feature" in field:
            result[field] = torch.tensor(x[field], dtype=None)
        else:
            result[field] = torch.tensor(x[field], dtype=type)
    return result


def concat_along_axis(x, axis):
    result = []
    swapped = zip(*x)
    print(x)
    for field in swapped:
        output = {}
        for entry in field[0]:
            output[entry] = np.concatenate([p[entry] for p in field],
                                           axis=axis)
        result.append(output)

    return tuple(result)


class Replay():
    def __init__(self, replay_path, replay_name, count):
        self.last_action = 0
        self.steps = count
        self.result = 0
        self.build = []
        self.last_obs = None
        self.replay_path = replay_path
        self.replay_name = os.path.join(replay_path, replay_name)
        self.replay_data = []
        self.discount = 1
        self.actions_list = []
        self.screen = []
        self.minimap = []
        self.units = []
        self.available_actions = []
        self.scores = []
        print(self.replay_path, self.replay_name)

    def save_info(self):
        inputs, target, masks = concat_along_axis(self.replay_data, 0)
        inputs = to_tensor(inputs)
        target = to_tensor(target, torch.int64)
        masks = to_tensor(masks)
        hidden = (torch.zeros(1, 8 * 8 * 2,
                              128), torch.zeros(1, 8 * 8 * 2, 128),
                  torch.zeros(2, 1, 1024),
                  torch.zeros(2, 1, 1024))
        hidden = tuple([hid.cuda() for hid in hidden])
        with open("/home/gilsson/replay_save/" + str(self.steps), "wb") as f:
            dump(compress(pickle.dumps((inputs, target, masks, hidden))), f)

    @ staticmethod
    def get_dict():
        result_dict = {}

        result_dict["function"] = np.zeros((1, 1), dtype=np.int32)
        result_dict["time_skip"] = np.zeros((1, 1), dtype=np.int32)

        for t in ac.TYPES:
            x = str(t)
            result_dict[x] = np.zeros((1, 1), dtype=np.int32)

        return result_dict

    @ staticmethod
    def get_action(raw_function):
        ''' return: function + args '''
        if raw_function is None:
            fun_id = 0
        else:
            fun_id: int = raw_function.function

        dict = Replay.get_dict()
        dict["function"][0][0] = fun_id
        dict["time_skip"][0][0] = 0
        if fun_id == 0:
            return fun_id, []

        for data, argt in zip(
                raw_function.arguments,
                ac.FUNCTION_TYPES[ac.FUNCTIONS[int(fun_id)].function_type]):
            id = data[0]
            shape = argt.sizes
            if len(shape) > 1:
                id = data[0] * 64 + data[1]

            name = str(argt)
            dict[name][0][0] = id

        return dict

    def step(self, obs, action, feature):

        if self.last_obs is None:
            self.last_obs = obs
            return

        if self.last_obs is not None:
            replay_action = None
            if len(action) > 0:
                act = action[0]
                try:
                    replay_action = feature.reverse_action(act)
                except Exception:
                    replay_action = None

            if replay_action:
                print(replay_action)
                rd, md = extract_action(replay_action)
                new_obs = obs
                func_name = replay_action.function.name

                if ("Train" in func_name
                        or "Build" in func_name) and len(self.build) < 20:
                    unit_type = None
                    for race in (units.Neutral, units.Protoss, units.Terran, units.Zerg):
                        try:
                            unit_type = race[func_name.split("_")[1]]
                        except KeyError:
                            pass
                    self.build.append(unit_type)

                if replay_action == 1:
                    if self.last_action == 1:
                        del self.replay_data[-1]

                self.replay_data.append((get_observation(
                    feature.transform_obs(self.last_obs)), rd, md))
                self.last_obs = new_obs
                self.last_action = replay_action.function

        else:
            self.last_obs = obs


def run_replay(names, state: Replay):
    set_num_threads(1)
    for i, name in enumerate(names):
        path = os.path.join(target_dir, os.path.basename(name))

        run_config: RunConfig = run_configs.get(version="latest")
        sc2_proc = run_config.start()
        controller = sc2_proc.controller
        replay_data = run_config.replay_data(name)

        ping = controller.ping()
        try:
            info = controller.replay_info(replay_data)
        except Exception:
            sc2_proc.close()
            continue
        player_id = None
        for p in info.player_info:
            player_id = p.player_info.player_id

        interface = sc2_env.SC2Env._get_interface(
            AGENT_INTERFACE_FORMAT, False)

        if info.local_map_path:
            map_data = run_config.map_data(info.local_map_path)

        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            realtime=False,
            options=interface,
            observed_player_id=player_id))

        feature = features.features_from_game_info(controller.game_info())

        # print(feature)

        episode_count = info.game_duration_loops
        for a in range(0, episode_count):
            controller.step(1)
            obs = controller.observe()
            state.step(obs, obs.actions, feature)
            if obs.player_result:
                break

        state.save_info()
        sc2_proc.close()
        continue


if __name__ == "__main__":
    FLAGS = flags.FLAGS
    import sys
    FLAGS(sys.argv)
    names = []
    directory = '/home/gilsson/Replay2/'
    target_dir = '/home/gilsson/StarTrain/model/'
    for path in os.listdir(directory):
        names.append(os.path.join(directory, path))

    NUM_WORKERS = 2
    split = np.array_split(names, NUM_WORKERS)
    print(split)
    for i, group in enumerate(split):
        state = Replay(directory, os.path.basename(group[0]), i)
        worker = Process(target=run_replay, args=(group, state))
        worker.start()
