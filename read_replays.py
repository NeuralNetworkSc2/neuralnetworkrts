from pysc2.env.environment import StepType
from pysc2.lib import features
from pysc2.lib import units
from pysc2 import run_configs
from pysc2.env import sc2_env
from pysc2.run_configs.lib import RunConfig
from s2clientprotocol import sc2api_pb2 as sc_pb, common_pb2
from torch import set_num_threads
import os
from torch.multiprocessing import Process
from pickle import dump, load
from pysc2.lib.remote_controller import RemoteController
import torch
from zstd import compress
from os import mkdir, path
from main import AGENT_INTERFACE_FORMAT
from absl import flags
import numpy as np
from pysc2.lib import actions as ac

screen_size = (64, 64)
minimap_size = (64, 64)
NUM_WORKERS = 2


class Replay():
    def __init__(self, replay_path, replay_name):
        self.last_action = 0
        self.steps = 0
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
        data = {"build": self.build, "result": self.result,
                "actions": self.actions_list, }
        print('save info')
        print(data)
        path = os.path.join("/home/gilsson/replay_save/",
                            os.path.basename(self.replay_name) + ".pkl")
        print(path)
        with open(path, "wb") as f:
            dump(data, f)

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

        if obs.player_result:
            step_type = StepType.LAST
            self.discount = 0

        agent_obs = feature.transform_obs(obs)

        # print(f"agent obs: {agent_obs}")

        if len(action) != 0:
            exec_actions = []
            for actions in action:
                # print('a')
                exec_act = feature.reverse_action(actions)

                func = int(exec_act.function)
                args = []
                for arg in exec_act.arguments:
                    if type(arg[0]).__name__ != 'int':
                        args.append(arg[0].value)
                    else:
                        args.append(arg)

                exec_actions.append([func, args])

                print(f"exec_actions: {exec_actions[-1]}")

            self.actions_list.append(exec_actions)
        if self.last_obs is not None:
            replay_action = None
            if len(action) > 0:
                act = action[0]
                try:
                    replay_action = feature.reverse_action(act)
                except Exception:
                    replay_action = None

            if replay_action:

                # print(replay_action)
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
                    print('builddddddddddddddddddddddddddddddd')
                    print(unit_type)
                    self.build.append(unit_type)

                if replay_action == 1:  # move camera
                    if self.last_action == 1:
                        del self.replay_data[-1]

                self.replay_data.append(feature.transform_obs(self.last_obs))
                self.last_obs = new_obs
                self.last_action = replay_action.function

        else:
            self.last_obs = obs
        self.steps += 1


# def valid_replay(info, ping):
#     if (info.HasField("error") or info.base_build != ping.base_build or
#             info.game_duration_loops < 1000 or len(info.player_info) != 2):
#         return False
#     return True


def run_replay(names, state: Replay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    set_num_threads(1)
    for i, name in enumerate(names):
        # print(i, name)
        path = os.path.join(target_dir, os.path.basename(name))

        run_config: RunConfig = run_configs.get(version="latest")
        print(run_config)
        sc2_proc = run_config.start()
        print(sc2_proc)
        controller = sc2_proc.controller
        replay_data = run_config.replay_data(name)

        ping = controller.ping()
        print(ping)
        try:
            info = controller.replay_info(replay_data)
            print('aboba1')
        except Exception:
            sc2_proc.close()
            continue
        player_id = None
        for p in info.player_info:
            player_id = p.player_info.player_id

        player_0_id = info.player_info[0].player_info.race_actual
        state.save_info()

        # if not valid_replay(info, ping):
        #     sc2_proc.close()
        #     continue

        # print(player_id)
        interface = sc2_env.SC2Env._get_interface(
            AGENT_INTERFACE_FORMAT, False)

        map_data = None

        print('replay info', info)
        if info.local_map_path:
            map_data = run_config.map_data(info.local_map_path)

        controller.start_replay(sc_pb.RequestStartReplay(
            replay_data=replay_data,
            realtime=True,
            options=interface,
            observed_player_id=player_id))

        feature = features.features_from_game_info(controller.game_info())

        print(feature)

        episode_count = info.game_duration_loops
        for _ in range(0, episode_count):
            controller.step(1)
            obs = controller.observe()
            state.step(obs, obs.actions, feature)
            if obs.player_result:
                break

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

    split = np.array_split(names, 1)
    for i, group in enumerate(split):
        state = Replay(directory, os.path.basename(group[i]))
        worker = Process(target=run_replay, args=(group, state))
        worker.start()
