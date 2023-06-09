from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features
from absl import app


class TerranAgent(base_agent.BaseAgent):
    def step(self, obs):
        super(TerranAgent, self).step(obs)

        return actions.FUNCTIONS.no_op()


def main(unused_argv):
    agent = TerranAgent()

    try:
        while True:
            with sc2_env.SC2Env(  # 创建SC2Env的同时将开启StarCraft2游戏进程
                    map_name="Acropolis",
                    players=[sc2_env.Agent(sc2_env.Race.protoss),
                             # sc2_env.Agent(sc2_env.Race.terran),
                             sc2_env.Bot(sc2_env.Race.random, sc2_env.Difficulty.very_easy)
                             ],
                    # 设定接口相关选项，选择你希望获取哪些信息（例如是否获取屏幕外的信息）
                    agent_interface_format=features.AgentInterfaceFormat(
                        use_raw_units=True,
                        use_unit_counts=True,
                        feature_dimensions=features.Dimensions(screen=84, minimap=64)),  # 设定屏幕和小地图的分辨率
                    step_mul=16,  # 游戏迭代次数 / 代理迭代次数
                    realtime=True,
                    game_steps_per_episode=0,
                    visualize=False,  # 另开窗口显示camera和feature层
                    window_size=(1500, 1200),  # 游戏窗口大小
                    disable_fog=True,  # 是否有战争迷雾
            ) as env:

                timesteps: tuple[sc2_env.environment.TimeStep]  # the type of timesteps
                timesteps = env.reset()

                agent.setup(env.observation_spec(), env.action_spec())
                agent.reset()

                while True:
                    step_actions = [agent.step(timesteps[0])]  # 这里只使用了一个agent
                    if timesteps[0].last():
                        break
                    timesteps = env.step(step_actions)  # timesteps就是传给agent的obs

                    if timesteps[0].observation.unit_counts.shape[0] != 2:
                        print('Find enemy!')

    except KeyboardInterrupt:
        exit(0)


if __name__ == "__main__":
    app.run(main)


