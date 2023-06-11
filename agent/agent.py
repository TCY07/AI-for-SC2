import random
import numpy as np
import pandas as pd
import os
from absl import app
from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from pysc2.env import sc2_env, run_loop


class QLearningTable:
    def __init__(self, actions, learning_rate=0.5, reward_decay=0.9):
        self.actions = actions
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.q_table = None

        self.load()

    def choose_action(self, state, greedy):
        self.check_state_exist(state)
        if greedy:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table.loc[state, :]
            action = np.random.choice(
                state_action[state_action == np.max(state_action)].index)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        q_target = r + self.reward_decay * self.q_table.loc[s_, :].max()

        self.q_table.loc[s, a] += self.learning_rate * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions),
                                                         index=self.q_table.columns,
                                                         name=state))

    def save(self):
        self.q_table.to_pickle('QTable.pkl')

    def load(self):
        if os.path.exists('QTable.pkl'):
            self.q_table = pd.read_pickle('QTable.pkl')

            # self.q_table.loc['(0, 1, 0, 0, 0, 0)', :] = [10, 0, 0, 0, 0, 0]
            # self.q_table.loc['(1, 1, 1, 0, 0, 0)', :] = [0, 10, 0, 0, 0, 0]
            # self.q_table.loc['(0, 1, 1, 0, 0, 0)', :] = [0, 0, 0, 10, 0, 0]
            # self.q_table.loc['(0, 1, 1, 1, 0, 0)', :] = [0, 0, 0, 10, 0, 0]
            # self.save()
            # print("done")

            print('Load QTable successful')
        else:
            print('Load QTable failed')
            self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)


class Agent(base_agent.BaseAgent):
    actions = ("do_nothing",
               "harvest_minerals",
               "build_supply_depot",
               "build_barracks",
               "train_marine",
               "attack")

    @staticmethod
    def get_my_units_by_type(obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.SELF]

    @staticmethod
    def get_enemy_units_by_type(obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.alliance == features.PlayerRelative.ENEMY]

    @staticmethod
    def get_my_completed_units_by_type(obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    @staticmethod
    def get_enemy_completed_units_by_type(obs, unit_type):
        return [unit for unit in obs.observation.raw_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.ENEMY]

    @staticmethod
    def get_distances(obs, units, xy):
        units_xy = [(unit.x, unit.y) for unit in units]
        return np.linalg.norm(np.array(units_xy) - np.array(xy), axis=1)

    def step(self, obs):
        super(Agent, self).step(obs)
        if obs.first():
            command_center = self.get_my_units_by_type(
                obs, units.Terran.CommandCenter)[0]
            self.base_top_left = (command_center.x < 32)

    def do_nothing(self, obs):
        return actions.RAW_FUNCTIONS.no_op()

    def harvest_minerals(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        if len(idle_scvs) > 0:
            mineral_patches = [unit for unit in obs.observation.raw_units
                               if unit.unit_type in [
                                   units.Neutral.BattleStationMineralField,
                                   units.Neutral.BattleStationMineralField750,
                                   units.Neutral.LabMineralField,
                                   units.Neutral.LabMineralField750,
                                   units.Neutral.MineralField,
                                   units.Neutral.MineralField750,
                                   units.Neutral.PurifierMineralField,
                                   units.Neutral.PurifierMineralField750,
                                   units.Neutral.PurifierRichMineralField,
                                   units.Neutral.PurifierRichMineralField750,
                                   units.Neutral.RichMineralField,
                                   units.Neutral.RichMineralField750
                               ]]
            actions_to_return = []
            for scv in scvs:
                distances = self.get_distances(obs, mineral_patches, (scv.x, scv.y))
                mineral_patch = mineral_patches[np.argmin(distances)]
                actions_to_return.append(actions.RAW_FUNCTIONS.Harvest_Gather_unit("now", scv.tag, mineral_patch.tag))
            return actions_to_return
        return None

    def build_supply_depot(self, obs):
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        # idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        available_scvs = [scv for scv in scvs if scv.order_id_0 == 359]  # 采矿的scv视为available

        if len(supply_depots) >= 3 or len(available_scvs) <= 0:
            return None

        if obs.observation.player.minerals >= 100:
            supply_depot_xy = (20 - 2 * len(supply_depots), 26) if self.base_top_left \
                else (37 + 2 * len(supply_depots), 42)
            scv = available_scvs[random.randint(0, len(available_scvs) - 1)]
            return actions.RAW_FUNCTIONS.Build_SupplyDepot_pt("now", scv.tag, supply_depot_xy)
        else:
            return actions.RAW_FUNCTIONS.no_op()

    def build_barracks(self, obs):
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        available_scvs = [scv for scv in scvs if scv.order_id_0 == 359]  # 采矿的scv视为available

        if len(completed_supply_depots) <= 0 or len(barrackses) >= 5 or len(available_scvs) == 0:
            return None

        if obs.observation.player.minerals >= 150:
            barracks_xy = (24, 18 + 2 * len(barrackses)) if self.base_top_left else (33, 50 - 2 * len(barrackses))
            scv = available_scvs[random.randint(0, len(available_scvs) - 1)]
            return actions.RAW_FUNCTIONS.Build_Barracks_pt("now", scv.tag, barracks_xy)
        return actions.RAW_FUNCTIONS.no_op()

    def train_marine(self, obs):
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)

        if len(completed_barrackses) > 0 and free_supply > 0:
            action_to_return = [actions.RAW_FUNCTIONS.no_op()]
            for barracks in completed_barrackses:
                if barracks.order_length < 5 and obs.observation.player.minerals >= 50:
                    action_to_return.append(actions.RAW_FUNCTIONS.Train_Marine_quick("now", barracks.tag))
            return action_to_return
        else:
            return None

    def attack(self, obs):
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)
        if len(marines) > 0:
            attack_xy = (38, 44) if self.base_top_left else (19, 23)
            x_offset = random.randint(-10, 10)
            y_offset = random.randint(-10, 10)
            return [actions.RAW_FUNCTIONS.Attack_pt(
                "now", marine.tag, (attack_xy[0] + x_offset, attack_xy[1] + y_offset)) for marine in marines]
        else:
            return None


class SmartAgent(Agent):
    def __init__(self):
        super(SmartAgent, self).__init__()
        self.qtable = QLearningTable(self.actions)
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None
        self.previous_value_units = None
        self.previous_killed_value = None

        self.new_game()

    def reset(self):
        super(SmartAgent, self).reset()
        self.new_game()

    def new_game(self):
        self.base_top_left = None
        self.previous_state = None
        self.previous_action = None

        self.previous_value_units = None
        self.previous_killed_value = None

    def get_state(self, obs):
        scvs = self.get_my_units_by_type(obs, units.Terran.SCV)
        idle_scvs = [scv for scv in scvs if scv.order_length == 0]
        command_centers = self.get_my_units_by_type(obs, units.Terran.CommandCenter)
        supply_depots = self.get_my_units_by_type(obs, units.Terran.SupplyDepot)
        completed_supply_depots = self.get_my_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        barrackses = self.get_my_units_by_type(obs, units.Terran.Barracks)
        completed_barrackses = self.get_my_completed_units_by_type(
            obs, units.Terran.Barracks)
        marines = self.get_my_units_by_type(obs, units.Terran.Marine)

        queued_marines = (completed_barrackses[0].order_length
                          if len(completed_barrackses) > 0 else 0)

        free_supply = (obs.observation.player.food_cap -
                       obs.observation.player.food_used)
        can_afford_supply_depot = obs.observation.player.minerals >= 100
        can_afford_barracks = obs.observation.player.minerals >= 150
        can_afford_marine = obs.observation.player.minerals >= 100

        enemy_scvs = self.get_enemy_units_by_type(obs, units.Terran.SCV)
        enemy_idle_scvs = [scv for scv in enemy_scvs if scv.order_length == 0]
        enemy_command_centers = self.get_enemy_units_by_type(
            obs, units.Terran.CommandCenter)
        enemy_supply_depots = self.get_enemy_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_completed_supply_depots = self.get_enemy_completed_units_by_type(
            obs, units.Terran.SupplyDepot)
        enemy_barrackses = self.get_enemy_units_by_type(obs, units.Terran.Barracks)
        enemy_completed_barrackses = self.get_enemy_completed_units_by_type(
            obs, units.Terran.Barracks)
        enemy_marines = self.get_enemy_units_by_type(obs, units.Terran.Marine)

        return (
            # len(command_centers),
            # len(scvs),
            len(idle_scvs),
            len(supply_depots),
            len(completed_supply_depots),
            len(barrackses),
            len(completed_barrackses),
            len(marines),
            # queued_marines,
            # free_supply,
            # can_afford_supply_depot,
            # can_afford_barracks,
            # can_afford_marine,
            # len(enemy_command_centers),
            # len(enemy_scvs),
            # len(enemy_idle_scvs),
            # len(enemy_supply_depots),
            # len(enemy_completed_supply_depots),
            # len(enemy_barrackses),
            # len(enemy_completed_barrackses),
            # len(enemy_marines)
        )

    def step(self, obs):
        super(SmartAgent, self).step(obs)
        state = str(self.get_state(obs))

        greedy = 0.5
        is_greedy = random.random() < greedy

        while True:
            action = self.qtable.choose_action(state, is_greedy)
            sc_actions = getattr(self, action)(obs)
            if sc_actions is not None:  # action不为None，表示对于当前状态而言可行（但可能有规定状态以外的因素影响，使其不能执行）
                print('choose action: ' + action)
                break

        killed_value_units = obs.observation['score_cumulative'][5]
        total_value_units = obs.observation['score_cumulative'][3]

        reward = 0

        if self.previous_action is not None:
            delta_value_units = total_value_units - self.previous_value_units
            reward += 0.001 * (killed_value_units + delta_value_units)

        self.previous_value_units = total_value_units
        self.previous_killed_value = killed_value_units

        if obs.last():
            reward += obs.reward  # 胜利/失败的奖励

        if self.previous_action is not None:
            self.qtable.learn(self.previous_state,
                              self.previous_action,
                              reward,
                              'terminal' if obs.last() else state)

        if obs.last():
            self.qtable.save()

        self.previous_state = state
        self.previous_action = action

        return sc_actions


def main(unused_argv):
    agent1 = SmartAgent()
    try:
        with sc2_env.SC2Env(
                map_name="Simple64",
                players=[sc2_env.Agent(sc2_env.Race.terran),
                         sc2_env.Bot(sc2_env.Race.terran, sc2_env.Difficulty.hard)],
                agent_interface_format=features.AgentInterfaceFormat(
                    action_space=actions.ActionSpace.RAW,
                    use_raw_units=True,
                    raw_resolution=64,
                ),
                step_mul=48,
                disable_fog=True,
                window_size=(1200, 800),  # 游戏窗口大小
        ) as env:
            run_loop.run_loop([agent1], env, max_episodes=1000)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app.run(main)
