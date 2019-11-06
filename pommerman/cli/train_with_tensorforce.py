"""Train an agent with TensorForce.

Call this with a config, a game, and a list of agents, one of which should be a
tensorforce agent. The script will start separate threads to operate the agents
and then report back the result.

An example with all three simple agents running ffa:
python train_with_tensorforce.py \
 --agents=tensorforce::ppo,test::agents.SimpleAgent,test::agents.SimpleAgent,test::agents.SimpleAgent \
 --config=PommeFFACompetition-v0
"""
import atexit
import functools
import os

import argparse
import docker
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym
import gym
import action_prune
from pommerman import helpers, make
from pommerman.agents import TensorForceAgent


CLIENT = docker.from_env()
RES_WIN = 0
RES_LOSE = 1
RES_DRAW = 2
ACT_SLEEP = 3
ACT_BOMB = 4
ACT_OTHER = 5
ITEM_BLAST = 6
ITEM_KICK = 7
ITEM_AMMO = 8

STR_WINNER='Winner' # :thumbs_up_light_skin_tone:'
STR_LOSER='Loser' # :thumbs_down_light_skin_tone:'
STR_SLEEP='Sleep'
STR_STAY='Stay'
STR_UP='Up'
STR_LEFT='Left'
STR_DOWN='Down'
STR_RIGHT='Right'
STR_BOMBSET='BombSet' # :bomb:'
STR_BLAST='ItemBlast' # :cookie:'
STR_KICK='ItemKick' # :egg:'
STR_AMMO='ItemAmmo' # :rice:'
GAME_STEP = 200

def clean_up_agents(agents):
    """Stops all agents"""
    return [agent.shutdown() for agent in agents]


class WrappedEnv(OpenAIGym):
    '''An Env Wrapper used to make it easier to work
    with multiple agents'''

    def __init__(self, gym, visualize=False):
        self.gym = gym
        self.visualize = visualize
        #追加
        self.old_position = None
        self.prev_position = None
        self.curr_position = None
        self.timestep = 0
        self.episode = 0
        self.has_blast_strength = False
        self.has_can_kick = False
        self.has_ammo = False
        self.tmp_reward = 0.0
        self.res_reward = 0.0
        self.accu_bombset = 1.0
        self.act_history = []
        self.render = False
        self.bombset_count = 0
        self.bombset_flag = 0
        self.lazy = 0.01
        #self.rewards = DEFAULT_REWARDS
        self.oldwallcount = 0
        self.old_observations = []
        print(f'Episode [{self.episode:03}], Timestep [{self.timestep:03}] initialized.')

    def tutorial_reward(self, agent_id, agent_obs, agent_reward, agent_action):
        import emoji
        import numpy as np
        self.timestep += 1
        self.agent_board = agent_obs['board']
        self.agent_obs_bomb = agent_obs['bomb_life']
        self.curr_position = np.where(self.agent_board == agent_id)
        self.tmp_reward = 0.0
        wallcount = 0
        actions = []
        tatebombs = [0,0,0,0,0,0,0,0,0,0,0]
        yokobombs = [0,0,0,0,0,0,0,0,0,0,0]
        bombcount = 0
        agent_pos = [1,3]
        #print(agent_obs.keys())
        #print(agent_obs['bomb_life'])
        #agent_oldobs[]
            #self.tmp_reward += 0.36
        for i in range(11):
            for j in range(11):
                if self.agent_board[i][j] == 2:
                    wallcount += 1
                if self.agent_obs_bomb[i][j] <= 3 and self.agent_obs_bomb[i][j] != 0:
                    tatebombs[bombcount] = i
                    yokobombs[bombcount] = j
                if self.agent_board[i][j] == 10:
                    agent_pos[0] = i
                    agent_pos[1] = j
                """
                if self.agent_board[i][j] == 10:
                    if tatebombs[bombcount] == i:
                        if yokobombs[bombcount] <= j+1 and yokobombs[bombcount] <= j-1
                            self.tmp_reward -= 0.1
                """

        #print(self.agent_board)
        #print(tatebombs,yokobombs)
        #爆弾の近くにいると減点
        if self.timestep == 1:
            self.old_wallcount = wallcount
            #self.tmp_reward += 0.10
        #if self.timestep == GAME_STEP:
            #self.tmp_reward += 1.5
            #print("コロンビア")
        #print('wallcount:',self.old_wallcount,wallcount)
        #print(agent_pos[0],tatebombs[bombcount])
        if agent_pos[0] == tatebombs[bombcount]:
            self.tmp_reward -= 0.15
        if agent_pos[1] == yokobombs[bombcount]:
            self.tmp_reward -= 0.15

        if self.old_wallcount != wallcount:
            self.tmp_reward += ((self.old_wallcount - wallcount)*0.04)

            #print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        if self.bombset_count <= 0:
            if self.bombset_flag >= 1:
                self.tmp_reward += 0.5 * self.bombset_flag
                self.bombset_flag = 0
            if agent_action == 5:
                self.bombset_count = 10
                self.bombset_flag += 1
            """
            elif self.bombset_count <= -10:
                self.tmp_reward -= 0.5 * self.bombset_flag
                self.lazy = -0.01
            """
            #else:
                #self.lazy += 0.01
        #else:
        self.bombset_count -= 1


        #self.tmp_reward -= self.lazy#100step生存で-1になる
        self.res_reward += self.tmp_reward
        #self.act_history += actions
        # アクションの履歴を保存
        if self.render:
            print(f'Episode [{self.episode:03}], Timestep [{self.timestep:03}] got reward {round(self.res_reward, 2)} [{actions}]')
        #print(self.old_wallcount)
        #print(wallcount)
        #壁の数を数える##############################################################
        self.old_wallcount = wallcount
        ####################################################################
        self.old_position = self.prev_position
        self.prev_position = self.curr_position
        return self.tmp_reward

    def execute(self, action):#ここのアクションがどこで作られるのか知りたい
        if self.visualize:
            self.gym.render()
        #print("execute(action)",action)
        actions = self.unflatten_action(action=action)#エージェントのアクション決定
        obs = self.gym.get_observations()
        all_actions = self.gym.act(obs)
        #print(self.gym.training_agent)#0
        all_actions.insert(self.gym.training_agent, actions)
        state, reward, terminal, _ = self.gym.step(all_actions)
        agent_state = self.gym.featurize(state[self.gym.training_agent])
        agent_reward = reward[self.gym.training_agent]
        agent_id = self.gym.training_agent + 10#追加
        agent_action = actions#追加
        #print(actions)
        agent_obs = obs[self.gym.training_agent]#追加
        modified_reward = self.tutorial_reward(agent_id, agent_obs, agent_reward, agent_action)#追加
        return agent_state, terminal, modified_reward#変更

    def reset(self):
        #追加###########################
        hist = self.act_history
        item_count = hist.count(STR_AMMO) + hist.count(STR_BLAST) + hist.count(STR_KICK)
        bomb_count = hist.count(STR_BOMBSET)
        move_count = hist.count(STR_UP) + hist.count(STR_DOWN) + hist.count(STR_LEFT) + hist.count(STR_RIGHT)
        stop_count = hist.count(STR_SLEEP) + hist.count(STR_STAY)
        #history = "BombSet({}), ItemGot({}), Move({}), Stay({})".format(bomb_count, item_count, move_count, stop_count)
        history = "{},{},{},{}".format(bomb_count, item_count, move_count, stop_count)
        print(f'Episode [{self.episode:03}], Timestep [{self.timestep:03}] reward {round(self.res_reward,3)}')
        #######################
        obs = self.gym.reset()
        agent_obs = self.gym.featurize(obs[3])
        #追加#######################
        self.timestep = 0
        self.episode += 1
        self.tmp_reward = 0.0
        self.res_reward = 0.0
        self.accu_bombset = 1.0
        self.act_history = []
        self.has_blast_strength = False
        self.has_can_kick = False
        self.has_ammo = False
        self.oldwallcount = 0
        self.bombset_count = 0
        self.bombset_flag = 0
        self.lazy = 0.01
        #############################
        return agent_obs


def main():
    '''CLI interface to bootstrap taining'''
    parser = argparse.ArgumentParser(description="Playground Flags.")
    parser.add_argument("--game", default="pommerman", help="Game to choose.")
    parser.add_argument(
        "--config",
        default="PommeFFACompetition-v0",
        help="Configuration to execute. See env_ids in "
        "configs.py for options.")
    parser.add_argument(
        "--agents",
        default="tensorforce::ppo,test::agents.SimpleAgent,"
        "test::agents.SimpleAgent,test::agents.SimpleAgent",
        help="Comma delineated list of agent types and docker "
        "locations to run the agents.")
    parser.add_argument(
        "--agent_env_vars",
        help="Comma delineated list of agent environment vars "
        "to pass to Docker. This is only for the Docker Agent."
        " An example is '0:foo=bar:baz=lar,3:foo=lam', which "
        "would send two arguments to Docker Agent 0 and one to"
        " Docker Agent 3.",
        default="")
    parser.add_argument(
        "--record_pngs_dir",
        default=None,
        help="Directory to record the PNGs of the game. "
        "Doesn't record if None.")
    parser.add_argument(
        "--record_json_dir",
        default=None,
        help="Directory to record the JSON representations of "
        "the game. Doesn't record if None.")
    parser.add_argument(
        "--render",
        default=False,
        action='store_true',
        help="Whether to render or not. Defaults to False.")
    parser.add_argument(
        "--game_state_file",
        default=None,
        help="File from which to load game state. Defaults to "
        "None.")
    args = parser.parse_args()

    config = args.config
    record_pngs_dir = args.record_pngs_dir
    record_json_dir = args.record_json_dir
    agent_env_vars = args.agent_env_vars
    game_state_file = args.game_state_file

    # TODO: After https://github.com/MultiAgentLearning/playground/pull/40
    #       this is still missing the docker_env_dict parsing for the agents.
    agents = [
        helpers.make_agent_from_string(agent_string, agent_id + 1000)
        for agent_id, agent_string in enumerate(args.agents.split(","))
    ]

    env = make(config, agents, game_state_file)
    training_agent = None

    for agent in agents:
        if type(agent) == TensorForceAgent:
            training_agent = agent
            env.set_training_agent(agent.agent_id)
            break

    if args.record_pngs_dir:
        assert not os.path.isdir(args.record_pngs_dir)
        os.makedirs(args.record_pngs_dir)
    if args.record_json_dir:
        assert not os.path.isdir(args.record_json_dir)
        os.makedirs(args.record_json_dir)

    # Create a Proximal Policy Optimization agent
    agent = training_agent.initialize(env)
    #agent.restore_model("./gakushu-models/940000/")
    atexit.register(functools.partial(clean_up_agents, agents))
    wrapped_env = WrappedEnv(env, visualize=args.render)
    runner = Runner(agent=agent, environment=wrapped_env)
    ##########################エピソード終了時の処理
    def episode_finished(r):
        #print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
        if r.episode % 20000 == 0:
            print("Finished episode {ep} after {ts} timesteps".format(ep=r.episode + 1, ts=r.timestep + 1))
            print("Episode reward: {}".format(r.episode_rewards[-1]))
            agent.model.save(directory="./gakushu-models/" + str(r.episode) + "/")
        return True
    ##########################
    runner.run(episodes=1000000, max_episode_timesteps=GAME_STEP, episode_finished = episode_finished)#学習を実行
    #print("Stats: ", runner.episode_rewards, runner.episode_timesteps,
          #runner.episode_times)

    try:
        runner.close()
    except AttributeError as e:
        pass


if __name__ == "__main__":
    main()
