from read_replays import get_observation
import pysc2.lib.actions as ac
import pysc2.lib.features as ft
import sys
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from absl import flags
from pysc2.env.environment import StepType, TimeStep
import rl_agent
from pysc2.lib import actions
from pysc2.env import run_loop
from rl_agent import compute_sl_loss, optim
import torch
import os
from pickle import loads, load
from zstd import decompress
from pysc2.env import sc2_env
FLAGS = flags.FLAGS

AGENT_INTERFACE_FORMAT = ft.AgentInterfaceFormat(
    feature_dimensions=ft.Dimensions((64, 64),
                                     (64, 64)),
    hide_specific_actions=True,
    use_feature_units=True,
    # use_raw_units=True,
    # use_raw_actions=True,
    max_raw_actions=512,
    max_selected_units=30,
    use_unit_counts=False,
    use_camera_position=True,
    show_cloaked=False,
    show_burrowed_shadows=False,
    show_placeholders=False,
    # hide_specific_actions=False,
    action_delay_fn=None,
)
"""Initializer.

    Args:
      feature_dimensions: Feature layer `Dimension`s. Either this or
          rgb_dimensions (or both) must be set.
      rgb_dimensions: RGB `Dimension`. Either this or feature_dimensions
          (or both) must be set.
      raw_resolution: Discretize the `raw_units` observation's x,y to this
          resolution. Default is the map_size.
      action_space: If you pass both feature and rgb sizes, then you must also
          specify which you want to use for your actions as an ActionSpace enum.
      camera_width_world_units: The width of your screen in world units. If your
          feature_dimensions.screen=(64, 48) and camera_width is 24, then each
          px represents 24 / 64 = 0.375 world units in each of x and y.
          It'll then represent a camera of size (24, 0.375 * 48) = (24, 18)
          world units.
      use_feature_units: Whether to include feature_unit observations.
      use_raw_units: Whether to include raw unit data in observations. This
          differs from feature_units because it includes units outside the
          screen and hidden units, and because unit positions are given in
          terms of world units instead of screen units.
      use_raw_actions: [bool] Whether to use raw actions as the interface.
          Same as specifying action_space=ActionSpace.RAW.
      max_raw_actions: [int] Maximum number of raw actions
      max_selected_units: [int] The maximum number of selected units in the
          raw interface.
      use_unit_counts: Whether to include unit_counts observation. Disabled by
          default since it gives information outside the visible area.
      use_camera_position: Whether to include the camera's position (in minimap
          coordinates) in the observations.
      show_cloaked: Whether to show limited information for cloaked units.
      show_burrowed_shadows: Whether to show limited information for burrowed
          units that leave a shadow on the ground (ie widow mines and moving
          roaches and infestors).
      show_placeholders: Whether to show buildings that are queued for
          construction.
      hide_specific_actions: [bool] Some actions (eg cancel) have many
          specific versions (cancel this building, cancel that spell) and can
          be represented in a more general form. If a specific action is
          available, the general will also be available. If you set
          `hide_specific_actions` to False, the specific versions will also be
          available, but if it's True, the specific ones will be hidden.
          Similarly, when transforming back, a specific action will be returned
          as the general action. This simplifies the action space, though can
          lead to some actions in replays not being exactly representable using
          only the general actions.
      action_delay_fn: A callable which when invoked returns a delay in game
          loops to apply to a requested action. Defaults to None, meaning no
          delays are added (actions will be executed on the next game loop,
          hence with the minimum delay of 1).
      send_observation_proto: Whether or not to send the raw observation
          response proto in the observations.
      crop_to_playable_area: Crop the feature layer minimap observations down
          from the full map area to just the playable area. Also improves the
          heightmap rendering.
      raw_crop_to_playable_area: Crop the raw units to the playable area. This
          means units will show up closer to the origin with less dead space
          around their valid locations.
      allow_cheating_layers: Show the unit types and potentially other cheating
          layers on the minimap.
      add_cargo_to_units: Whether to add the units that are currently in cargo
          to the feature_units and raw_units lists.

    Raises:
      ValueError: if the parameters are inconsistent.
    """
if __name__ == "__main__":
    FLAGS(sys.argv)
    import sys
    FLAGS(sys.argv)
    # AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
    #     feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
    #     use_feature_units=True, use_raw_units=True,  rgb_dimensions=None
    # )
    # run_config = run_configs.get()
    # # with run_config.start() as controller:
    # while True:
    with sc2_env.SC2Env(
            map_name="Acropolis",
            players=[sc2_env.Agent(sc2_env.Race.protoss), sc2_env.Bot(
                sc2_env.Race.terran, sc2_env.Difficulty.easy)],
            realtime=True,
            agent_interface_format=AGENT_INTERFACE_FORMAT) as env:
        # net = rl_agent.Model(env.observation_spec()[
        #     0]['feature_screen'], 573)

        training = False
        net = rl_agent.Model(training).cuda()
        # print(env.observation_spec()[0]['feature_screen'])

        # agent = ptan.agent.PolicyAgent(lambda x: net(
        #     x)[0], apply_softmax=True, device=torch.device("cuda"))
        # exp_source = ptan.experience.ExperienceSourceFirstLast(env, agetn)
        agent = rl_agent.A3CAgent(net)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005, eps=1e-4)
        scaler = GradScaler()
        # run_loop.run_loop([agent], env, 100000)
        # action = [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]
        # obs = env.step(action)
        state_dict_path = "/home/gilsson/PythonProjects/neural-network-rts/sl_model.tm"
        checkpoint = torch.load(state_dict_path)

        # net.load_state_dict(checkpoint["model_state_dict"])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        last_hidden = []
        PATH = "/home/gilsson/replay_save/"
        for file in os.listdir(PATH):
            print(file)
            replay_path = os.path.join(PATH, file)
            replays = []
            count = 0

            for batch in sorted(os.listdir(replay_path)):
                with open(os.path.join(replay_path, batch), "rb") as steps:
                    replays.append(loads(decompress(load(steps))))

            while True:
                count += 1
                if len(last_hidden) == 0:
                    hiddens = (torch.zeros(1, 8 * 8 * 2,
                                           128), torch.zeros(1,  8 * 8 * 2, 128),
                               torch.zeros(1, 1, 128),
                               torch.zeros(1, 1, 128))
                else:
                    hiddens = last_hidden
                if training:

                    current_replay = replays[count]
                    inputs, targets, masks = zip(current_replay)
                    current_replay = (inputs[:50], targets[:50], masks[:50])

                    def concat(x):
                        output = {}
                        for entry in x[0]:
                            output[entry] = torch.cat(
                                [p[entry].cuda() for p in x], axis=1)

                        return output

                    def concat_lstm_hidden(x):
                        result = tuple()
                        swapped = zip(*x)
                        for field in swapped:
                            output = torch.cat(field, axis=1)
                            result = result + (output, )

                        return result
                    inputs = concat(current_replay[0])
                    targets = concat(current_replay[2])
                    masks = concat(current_replay[1])

                    for input in inputs:
                        inputs[input] = inputs[input].cuda()

                    for input in targets:
                        targets[input] = targets[input].cuda()

                    for input in masks:
                        masks[input] = masks[input].cuda()

                    with autocast():
                        out, new_hid = net(inputs, hiddens, targets)
                        loss, losses_dict, scores_dict = compute_sl_loss(
                            out, targets, masks)

                    reduced_loss = loss / 1 / 32
                    # if count % 4 == 0:
                    scaler.scale(reduced_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    last_hidden = [hidden.detach().cuda()
                                   for hidden in new_hid]

                    torch.save({
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    },  state_dict_path)
                    print('SAVED')

                else:
                    obs = env.step([actions.FunctionCall(0, [])])
                    time_step = obs[0]
                    inputs = get_observation(time_step.observation)
                    for input in inputs:
                        inputs[input] = torch.FloatTensor(inputs[input]).cuda()

                    out, new_hid = net(inputs, hiddens)
                    last_hidden = [hidden.detach().cuda()
                                   for hidden in new_hid]
                    action = out["function_sampled"].squeeze().item()
                    skip_target = max(
                        out["time_skip_sampled"].squeeze().item() - 1,
                        0) + time_step.observation["game_loop"][0]

                    func = ac.FUNCTIONS[action]
                    action_data = []
                    last_action = {}
                    for x in out:
                        last_action[x.replace("_sampled",
                                              "")] = out[x].detach().cpu()

                    for x in ac.FUNCTION_TYPES[func.function_type]:
                        sub_action = out[str(x) + "_sampled"].squeeze().item()

                        if "screen" in str(x) or "minimap" in str(x):
                            sub_action = (sub_action // 64, sub_action % 64)
                        else:
                            sub_action = (sub_action, )
                        action_data.append(sub_action)

                    func = ac.FunctionCall(action, action_data)
                    print(time_step.observation['available_actions'])
                    print(func)
                    agent.step(time_step, func)
