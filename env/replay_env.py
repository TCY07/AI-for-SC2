from pysc2.env.environment import Base, StepType, TimeStep
from pysc2 import run_configs
from pysc2.lib import replay, features

from s2clientprotocol import sc2api_pb2 as sc_pb
import copy


class ReplayEnv(Base):
    def __init__(self,
                 interface_format,
                 sc2_replay: str,  # replay file path
                 disable_fog=True,
                 observed_player=1,
                 full_screen=False,
                 window_size="640,480",
                 step_mul=1,  # 游戏迭代次数 / 代理迭代次数
                 ):
        self._full_screen = full_screen
        self._window_size = window_size
        self._step_mul = step_mul

        run_config = run_configs.get()
        self._interface_format = interface_format
        self._interface_options = self._get_interface(interface_format)

        self._replay_data = run_config.replay_data(sc2_replay)
        self._start_replay = sc_pb.RequestStartReplay(
            replay_data=self._replay_data,
            options=self._interface_options,
            disable_fog=disable_fog,
            observed_player_id=observed_player)
        version = replay.get_replay_version(self._replay_data)
        self._run_config = run_configs.get(version=version)  # Replace the run config.

        self._sc_process = None
        self._controller = None
        self._features = None
        self._state = StepType.FIRST

    def reset(self):
        self._sc_process = self._run_config.start(
            full_screen=self._full_screen,
            window_size=self._window_size,
            want_rgb=self._interface_options.HasField("render"))
        self._controller = self._sc_process.controller

        info = self._controller.replay_info(self._replay_data)
        print(" Replay info ".center(60, "-"))
        print(info)
        print("-" * 60)

        self._controller.start_replay(self._start_replay)
        self._state = StepType.FIRST

        self._features = features.features_from_game_info(
            game_info=self._controller.game_info(),
            agent_interface_format=self._interface_format)
        obs = self._controller.observe()
        agent_obs = self._features.transform_obs(obs)  # 转换observations的形式

        return tuple(TimeStep(
            step_type=self._state,
            reward=0,
            discount=0,
            observation=agent_obs) for _ in range(1))

    def step(self, action):
        self._controller.step(self._step_mul)
        obs = self._controller.observe()
        agent_obs = self._features.transform_obs(obs)  # 转换observations的形式

        if obs.player_result:  # Episode over.
            self._state = StepType.LAST
        else:
            self._state = StepType.MID

        return tuple(TimeStep(
            step_type=self._state,
            reward=0,
            discount=0,
            observation=agent_obs) for _ in range(1))

    @staticmethod
    def _get_interface(interface_format):
        if isinstance(interface_format, sc_pb.InterfaceOptions):
            if not interface_format.raw:
                interface_options = copy.deepcopy(interface_format)
                interface_options.raw = True
                return interface_options
            else:
                return interface_format

        aif = interface_format
        interface = sc_pb.InterfaceOptions(
            raw=True,
            show_cloaked=aif.show_cloaked,
            show_burrowed_shadows=aif.show_burrowed_shadows,
            show_placeholders=aif.show_placeholders,
            raw_affects_selection=True,
            raw_crop_to_playable_area=aif.raw_crop_to_playable_area,
            score=True)

        if aif.feature_dimensions:
            interface.feature_layer.width = aif.camera_width_world_units
            aif.feature_dimensions.screen.assign_to(
                interface.feature_layer.resolution)
            aif.feature_dimensions.minimap.assign_to(
                interface.feature_layer.minimap_resolution)
            interface.feature_layer.crop_to_playable_area = aif.crop_to_playable_area
            interface.feature_layer.allow_cheating_layers = aif.allow_cheating_layers

        if aif.rgb_dimensions:
            aif.rgb_dimensions.screen.assign_to(interface.render.resolution)
            aif.rgb_dimensions.minimap.assign_to(interface.render.minimap_resolution)

        return interface

    def observation_spec(self):
        pass

    def action_spec(self):
        pass
