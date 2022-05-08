import time
import numpy as np
import multiagent.scenarios as scenarios
from multiagent.environment import MultiAgentEnv
from model import collaboration_trainers, behavior_trainers
import time
import torch
import argparse
from tensorboardX import SummaryWriter

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
time_now = time.strftime('%y%m_%d%H%M')

def make_env(scenario_name, benchmark=False):
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def run(arglist):
    """step1: create the environment """
    env = make_env(arglist.scenario_name, arglist.benchmark)
    """step2: create agents"""
    obs = env.reset()
    obs_shape_n = [obs[i].shape[0] for i in range(env.n)]

    action_shape_n = [env.action_space[i].n for i in range(env.n)] # no need for stop bit
    num_adversaries = min(env.n, arglist.num_adversaries)
    collaboration_model = collaboration_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist)
    behavior_model = behavior_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist)
    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    agent_info = [[[]]] # placeholder for benchmarking info
    episode_rewards = [0.0] # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)] # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape 
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a


    print('starting iterations ...')
    obs_n = env.reset()

    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        if game_step > 1 and game_step % 100 == 0:   
            mean_agents_r = [round(np.mean(agent_rewards[idx][-200:-1]), 2) for idx in range(env.n)]
            mean_ep_r = round(np.mean(episode_rewards[-200:-1]), 3)
            print(" "*43 + 'episode reward:{} agents mean reward:{}'.format(mean_ep_r, mean_agents_r), end='\r')
        print('=Training: steps:{} episode:{}'.format(game_step, episode_gone), end='\r')
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action

            c_num, b_n = collaboration_model.interaction(obs_n) #c_num 最大的值， b_n所有交互值

            action_n = behavior_model.step(obs_n, obs_n[c_num])

            # interact with env
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            #env.render()
            # save the experience

            collaboration_model.memory.push(obs_n, b_n, new_obs_n, rew_n, done_n)
            behavior_model.memory.push(obs_n, obs_n[c_num], action_n, new_obs_n, new_obs_n[c_num], rew_n, done_n)
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            # train our agents 
            behavior_model.train(game_step)
            collaboration_model.train(game_step)

            # update the obs_n
            game_step += 1
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len-1)
            if done or terminal:
                collaboration_model.reset_rnn()
                episode_step = 0
                obs_n = env.reset()
                agent_info.append([[]])
                episode_rewards.append(0)
                for a_r in agent_rewards:   
                    a_r.append(0)
                continue

if __name__ == '__main__':
    parser = argparse.ArgumentParser("reinforcement learning experiments for multiagent environments")
    # environment
    parser.add_argument("--scenario_name", type=str, default="simple_spread", help="name of the scenario script")
    parser.add_argument("--start_time", type=str, default=time_now, help="the time when start the game")
    parser.add_argument("--per_episode_max_len", type=int, default=45, help="maximum episode length")
    parser.add_argument("--max_episode", type=int, default=150000, help="maximum episode length")
    parser.add_argument("--num-adversaries", type=int, default=1, help="number of adversaries")

    # core training parameters
    parser.add_argument("--device", default=device, help="torch device ")
    parser.add_argument("--learning_start_step", type=int, default=2000, help="learning start steps")
    parser.add_argument("--max_grad_norm", type=float, default=0.5, help="max gradient norm for clip")
    parser.add_argument("--learning_fre", type=int, default=100, help="learning frequency")
    parser.add_argument("--tao", type=int, default=0.01, help="how depth we exchange the par of the nn")
    parser.add_argument("--lr_a", type=float, default=1e-2, help="learning rate for adam optimizer")
    parser.add_argument("--lr_c", type=float, default=1e-2, help="learning rate for adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.97, help="discount factor")
    parser.add_argument("--batch_size", type=int, default=1256, help="number of episodes to optimize at the same time")
    parser.add_argument("--ta", type=int, default=128, help="number of time abstract")
    parser.add_argument("--memory_size", type=int, default=100000, help="number of data stored in the memory")
    parser.add_argument("--kx", type=float, default=100, help="value add to sample weight")
    parser.add_argument("--if_weighted_sample", type=bool, default=True, help="value add to sample weight")
    parser.add_argument("--num_units_1", type=int, default=128, help="number of units in the mlp")
    parser.add_argument("--num_units_2", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--num_units_openai", type=int, default=64, help="number of units in the mlp")

    # checkpointing
    parser.add_argument("--fre4save_model", type=int, default=400,
                        help="the number of the episode for saving the model")
    parser.add_argument("--start_save_model", type=int, default=400,
                        help="the number of the episode for saving the model")
    parser.add_argument("--save_dir", type=str, default="models", \
                        help="directory in which training state and model should be saved")
    parser.add_argument("--old_model_name", type=str, default="models/1911_122134_20000/", \
                        help="directory in which training state and model are loaded")

    # evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", \
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", \
                        help="directory where plot data is saved")

    arglist = parser.parse_args()
    run(arglist)
