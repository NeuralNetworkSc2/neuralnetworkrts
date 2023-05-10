import ptan
import random
import numpy as np
from pysc2.env import sc2_env
from pysc2 import run_configs
import pysc2.lib.actions as ac
import pysc2.lib.features as ft
import sys
from absl import flags
from pysc2.env.environment import TimeStep
from pysc2.lib.features import Dimensions
import rl_agent
import ptan
from pysc2.lib import actions
from pysc2.env import run_loop
import torch
import loader
import util
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

    # AGENT_INTERFACE_FORMAT = sc2_env.AgentInterfaceFormat(
    #     feature_dimensions=sc2_env.Dimensions(screen=64, minimap=64),
    #     use_feature_units=True, use_raw_units=True,  rgb_dimensions=None
    # )
    # run_config = run_configs.get()
    # # with run_config.start() as controller:
    with sc2_env.SC2Env(
            map_name="Acropolis",
            players=[sc2_env.Agent(sc2_env.Race.zerg), sc2_env.Bot(
                sc2_env.Race.terran, sc2_env.Difficulty.easy)],
            realtime=False,
            visualize=False,
            agent_interface_format=AGENT_INTERFACE_FORMAT) as env:
        net = rl_agent.Model(env.observation_spec()[
                             0]['feature_screen'], 573)

        # agent = ptan.agent.PolicyAgent(lambda x: net(
        #     x)[0], apply_softmax=True, device=torch.device("cuda"))
        # exp_source = ptan.experience.ExperienceSourceFirstLast(env, agetn)
        agent = rl_agent.A3CAgent(net)
        run_loop.run_loop([agent], env, 100000)
        # done = False
        # action = [actions.FunctionCall(actions.FUNCTIONS.no_op.id, [])]
        # obs = env.step(action)
        # while not done:
        #     print("observation::::::::::::::::::::::::::::::")
        #     print(obs[0].observation.feature_screen.player_relative)
        #     # print("observation_spec>>")
        #     # print(env.observation_spec())
        #     # print('---------------------------------------------')
        #     # print("action_spec>>")
        #     # print(env.action_spec())
        #     # print(random.choice(env.action_spec()[0][0]))
        #     action = np.random.choice(obs[0].observation.available_actions)
        #     args = [[np.random.randint(0, size) for size in arg.sizes]
        #             for arg in env.action_spec()[0][1][action].args]
        #     # print("args>>")
        #     obs = env.step(
        #         actions=[actions.FunctionCall(action, args)])
        #     # if timesteps[0].last():
        #     #     done = True
        #     print('---------------------------------------------')
        #     # nn = rl_agent.ActorCriticModel([64, 64], len(env.action_spec()))
