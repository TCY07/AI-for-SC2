from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app

from env.replay_env import ReplayEnv

import numpy as np


def main(unused_argv):

    with ReplayEnv(
            interface_format=features.AgentInterfaceFormat(
                use_raw_units=True,
                use_unit_counts=True,
                feature_dimensions=features.Dimensions(screen=84, minimap=64)),  # 设定屏幕和小地图的分辨率
            sc2_replay=r"C:\Users\TCY\Desktop\test.SC2Replay",
            observed_player=1,
            step_mul=32,
    ) as env:
        timesteps: tuple[sc2_env.environment.TimeStep]  # the type of timesteps
        timesteps = env.reset()

        while True:
            if timesteps[0].last():
                break
            timesteps = env.step(None)  # timesteps就是传给agent的obs

            obs = timesteps[0].observation

            # print(obs['game_loop'])
            a = np.array(obs['game_loop']).reshape(-1)

            # print(obs['unit_counts'])
            a = np.concatenate((a, np.array(obs['unit_counts']).reshape(-1)), axis=0)

            # print(obs['feature_minimap'])
            # feature_minimap:
            # ['height_map', 'visibility_map', 'creep', 'camera', 'player_id',
            # 'player_relative', 'selected', 'unit_type', 'alerts', 'pathable', 'buildable']
            a = np.concatenate((a, np.array(obs['feature_minimap']).reshape(-1)), axis=0)

            # print(obs['player'])
            # player:
            # ['player_id', 'minerals', 'vespene', 'food_used', 'food_cap', 'food_army',
            # 'food_workers', 'idle_worker_count', 'army_count', 'warp_gate_count', 'larva_count']
            a = np.concatenate((a, np.array(obs['player']).reshape(-1)), axis=0)

            print(a.shape)
            print('-' * 40)

    # except KeyboardInterrupt:
    #     exit(0)


if __name__ == "__main__":
    app.run(main)
