from pysc2.env import sc2_env
from pysc2.lib import features
from absl import app

from env.replay_env import ReplayEnv

import numpy as np
import os

ROOT = r"C:\Users\TCY\Documents\StarCraft II\Accounts\1126365997\2-S2-1-10376425\Replays\Multiplayer\\"
FILE_NAMES = [
    # "Dragon Scales LE (16)",
    # "Royal Blood LE (10)",
    # "Royal Blood LE (9)",
    # "Dragon Scales LE (15)",
    # "Dragon Scales LE (14)",

    "Gresvan LE (11)",
    "Ancient Cistern LE (14)",
    "Dragon Scales LE (13)",
    "Ancient Cistern LE (13)",
    "Gresvan LE (10)",
    "Royal Blood LE (8)",

    "Altitude LE (12)",
    "NeoHumanity LE (18)",
    "NeoHumanity LE (17)",
    "Babylon LE (14)",
    "Gresvan LE (9)",
    "Ancient Cistern LE (12)",
    "NeoHumanity LE (16)",
    "Dragon Scales LE (12)",
    "Ancient Cistern LE (11)",
    "Gresvan LE (8)",
    "NeoHumanity LE (15)",
]
debug = False


def main(unused_argv):
    if debug:
        generate_data()
        return

    for file in FILE_NAMES:
        generate_data(ROOT + file + '.SC2Replay', file)


def generate_data(*args):
    try:
        with ReplayEnv(
                interface_format=features.AgentInterfaceFormat(
                    use_raw_units=True,
                    use_unit_counts=True,
                    feature_dimensions=features.Dimensions(screen=84, minimap=64)),  # 设定屏幕和小地图的分辨率
                sc2_replay=(args[0] if len(args) else r"C:\Users\TCY\Desktop\test.SC2Replay"),
                observed_player=1,
                step_mul=200,
        ) as env1, \
                ReplayEnv(
                    interface_format=features.AgentInterfaceFormat(
                        use_raw_units=True,
                        use_unit_counts=True,
                        feature_dimensions=features.Dimensions(screen=84, minimap=64)),  # 设定屏幕和小地图的分辨率
                    sc2_replay=(args[0] if len(args) else r"C:\Users\TCY\Desktop\test.SC2Replay"),
                    observed_player=2,
                    step_mul=200,
                ) as env2, \
                open('./output/' + (args[1] + '_player1.txt' if len(args) > 1 else 'test_player1.txt'), 'w') as f1, \
                open('./output/' + (args[1] + '_player2.txt' if len(args) > 1 else 'test_player2.txt'), 'w') as f2:
            timesteps1: tuple[sc2_env.environment.TimeStep]  # the type of timesteps
            timesteps1, info1 = env1.reset()
            timesteps2: tuple[sc2_env.environment.TimeStep]  # the type of timesteps
            timesteps2, _ = env2.reset()

            print(info1)

            # result:
            # Victory = 1
            # Defeat = 2
            # Tie = 3
            # Undecided = 4
            winner = 1 if (info1.player_info[0].player_result.player_id == 1
                           and info1.player_info[0].player_result.result == 1) else 2

            if info1.player_info[0].player_result.result == 3 or info1.player_info[0].player_result.result == 4:
                winner = 0  # no winner

            f1.write(str(winner) + ' ' + str(info1.game_duration_loops) + '\n')
            f2.write(str(winner) + ' ' + str(info1.game_duration_loops) + '\n')

            while True:
                if timesteps1[0].last() or timesteps2[0].last():
                    break

                if timesteps1[0].observation['game_loop'] > 20000:
                    break

                timesteps1 = env1.step(None)  # timesteps就是传给agent的obs
                timesteps2 = env2.step(None)

                obs1 = timesteps1[0].observation
                obs2 = timesteps2[0].observation

                player1_data = np.array(obs1['game_loop']).reshape(-1)
                player2_data = np.array(obs2['game_loop']).reshape(-1)

                # print(obs1['unit_counts'])
                player1_data = np.concatenate((player1_data, np.array(obs1['unit_counts']).reshape(-1)), axis=0)
                player2_data = np.concatenate((player2_data, np.array(obs2['unit_counts']).reshape(-1)), axis=0)

                # print(obs1['feature_minimap'])
                # feature_minimap:
                # ['height_map', 'visibility_map', 'creep', 'camera', 'player_id',
                # 'player_relative', 'selected', 'unit_type', 'alerts', 'pathable', 'buildable']
                # player1_data = np.concatenate((player1_data, np.array(obs1['feature_minimap']).reshape(-1)), axis=0)

                # print(obs1['player'])
                # player:
                # ['player_id', 'minerals', 'vespene', 'food_used', 'food_cap', 'food_army',
                # 'food_workers', 'idle_worker_count', 'army_count', 'warp_gate_count', 'larva_count']
                # player1_data = np.concatenate((player1_data, np.array(obs1['player']).reshape(-1)), axis=0)

                # raw_units:
                # (look up in pysc2.features.FeatureUnit)
                player1_data = np.concatenate((player1_data, np.array(obs1['raw_units']).reshape(-1)), axis=0)
                player2_data = np.concatenate((player2_data, np.array(obs2['raw_units']).reshape(-1)), axis=0)

                f1.write(' '.join(str(d) for d in player1_data))
                f1.write('\n')
                f2.write(' '.join(str(d) for d in player2_data))
                f2.write('\n')

                print(player1_data.shape, player2_data.shape)
                print('-' * 40)

    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    # print(os.path.split(r"C:\Users\TCY\Desktop\test.SC2Replay")[1].split('.')[0])
    # exit(0)
    app.run(main)
