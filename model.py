import torch
import torch.nn as nn
import torch.optim as optim
from replay_buffer import ReplayBuffer,TimeAbstractMemory
from net import openai_actor, openai_critic, noisy_RNN, SelfAttention, collaboration_actor
from collections import namedtuple
import numpy as np
def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key]*tao + \
                    (1-tao)*state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


class collaboration_trainers():
    def __init__(self, env, num_adversaries, obs_shape_n, action_shape_n, arglist):
        """
        init the trainers or load the old model
        """
        self.arglist = arglist
        self.batch = self.arglist.batch_size // self.arglist.ta
        Transition = namedtuple('Transition',
                                ('state', 'b_n', 'next_state', 'reward', 'done'))
        self.memory = TimeAbstractMemory(arglist.memory_size//arglist.ta, arglist.ta, arglist.kx, arglist.gamma, transition=Transition)
        self.rnn_cur = [None for _ in range(env.n)]
        self.actors_cur = [None for _ in range(env.n)]
        self.critics_cur = [None for _ in range(env.n)]

        self.rnn_tar = [None for _ in range(env.n)]
        self.actors_tar = [None for _ in range(env.n)]
        self.critics_tar = [None for _ in range(env.n)]

        self.optimizers_rnn = [None for _ in range(env.n)]
        self.optimizers_ac = [None for _ in range(env.n)]
        self.optimizers_cr = [None for _ in range(env.n)]
        self.update_cnt = 0
        self.n = env.n

        self.input_size_global = sum(obs_shape_n) + sum(action_shape_n)

        # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
        for i in range(env.n):
            self.rnn_cur[i] = noisy_RNN(obs_shape_n[i], arglist).to(arglist.device)
            self.actors_cur[i] = collaboration_actor(arglist.num_units_openai, 1, arglist).to(arglist.device)
            self.critics_cur[i] = openai_critic(sum(obs_shape_n), env.n, arglist).to(arglist.device)

            self.rnn_tar[i] = noisy_RNN(obs_shape_n[i], arglist).to(arglist.device)
            self.actors_tar[i] = collaboration_actor(arglist.num_units_openai, 1, arglist).to(arglist.device)
            self.critics_tar[i] = openai_critic(sum(obs_shape_n), env.n, arglist).to(arglist.device)

            self.optimizers_rnn[i] = optim.Adam(self.rnn_cur[i].parameters(), arglist.lr_a)
            self.optimizers_ac[i] = optim.Adam(self.actors_cur[i].parameters(), arglist.lr_a)
            self.optimizers_cr[i] = optim.Adam(self.critics_cur[i].parameters(), arglist.lr_c)

        update_trainers(self.actors_cur, self.actors_tar, 1.0)  # update the target par using the cur
        update_trainers(self.critics_cur, self.critics_tar, 1.0)  # update the target par using the cur
        update_trainers(self.rnn_cur, self.rnn_tar, 1.0)

    def interaction(self, obs_n):
        ht_n = [rnn(torch.from_numpy(obs).to(self.arglist.device, torch.float).unsqueeze(0)).detach() \
                    for rnn, obs in zip(self.rnn_cur, obs_n)]
        b_n = torch.cat([ac(ht).detach() for ac, ht in zip(self.actors_cur, ht_n)], 1)
        c_num = torch.max(b_n, 1)[1]
        return c_num, b_n.squeeze(0).detach().cpu().numpy()

    def reset_rnn(self):
        for i in range(self.n):
            self.rnn_cur[i].init_gru_state()
            self.rnn_tar[i].init_gru_state()

    def train(self, game_step):
        """
            use this func to make the "main" func clean
            par:
            |input: the data for training
            |output: the data for next update
            """
        # update all trainers, if not in display or benchmark mode
        if game_step > self.arglist.learning_start_step and \
                (game_step - self.arglist.learning_start_step) % self.arglist.learning_fre == 0:
            if self.update_cnt == 0: print('\r=start training ...' + ' ' * 100)
            # update the target par using the cur
            self.update_cnt += 1

            if self.arglist.if_weighted_sample:
                all_transitions = self.memory.weighted_sample(self.batch)
            else:
                all_transitions = self.memory.sample(self.batch)
            for i in range(self.batch):
                transitions = all_transitions[i].memory
                batch = self.memory.transition(*zip(*transitions))  # 转换成为一批次
                obs = torch.tensor(batch.state, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)
                b = torch.tensor(batch.b_n, dtype=torch.float, device=self.arglist.device)
                next_obs = torch.tensor(batch.next_state, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)
                rews = torch.tensor(batch.reward, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)
                dones = torch.tensor(batch.reward, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)

                critic_ips = torch.tensor(batch.state, dtype=torch.float, device=self.arglist.device).transpose(0, 1).reshape(self.arglist.ta, -1)
                tar_b = []

                # update every agent in different memory batch
                for agent_idx in range(self.n):
                    self.reset_rnn()
                    obs_n_o = obs[agent_idx].squeeze(0)
                    obs_n_n = next_obs[agent_idx].squeeze(0)

                    # --use the data to update the ACTOR
                    # There is no need to cal other agent's action
                    h = self.rnn_cur[agent_idx](obs_n_o, ifbatch=True)
                    bi = self.actors_cur[agent_idx](h, model_original_out=True)
                    temp_b = b.clone()
                    temp_b[:, agent_idx:agent_idx + 1] = bi
                    # update the aciton of this agent
                    loss_a = torch.mul(-1, torch.mean(self.critics_cur[agent_idx](critic_ips, temp_b)))

                    self.optimizers_ac[agent_idx].zero_grad()
                    self.optimizers_rnn[agent_idx].zero_grad()
                    loss_a.backward()
                    nn.utils.clip_grad_norm_(self.actors_cur[agent_idx].parameters(), self.arglist.max_grad_norm)
                    self.optimizers_ac[agent_idx].step()
                    self.optimizers_rnn[agent_idx].step()

                    self.reset_rnn()
                    h_next = self.rnn_tar[agent_idx](obs_n_n, ifbatch=True)
                    bi_next = self.actors_tar[agent_idx](h_next, model_original_out=True)
                    tar_b.append(bi_next.detach())

                critic_ips_next = torch.tensor(batch.next_state, dtype=torch.float,device=self.arglist.device).transpose(0, 1).reshape(self.arglist.ta, -1)
                tar_b = torch.cat(tar_b, 1)

                for agent_idx in range(self.n):
                    self.reset_rnn()
                    rew = rews[agent_idx].squeeze(0)  # set the rew to gpu
                    done_n = dones[agent_idx].squeeze(0)  # set the rew to gpu

                    # --use the data to update the ACTOR
                    # There is no need to cal other agent's action

                    q = self.critics_cur[agent_idx](critic_ips, b).reshape(-1)  # q
                    q_ = self.critics_tar[agent_idx](critic_ips_next, tar_b).detach().reshape(-1)  # q_
                    tar_value = q_ * self.arglist.gamma * done_n + rew  # q_*gamma*done + reward
                    loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
                    self.optimizers_cr[agent_idx].zero_grad()
                    loss_c.backward()
                    nn.utils.clip_grad_norm_(self.critics_cur[agent_idx].parameters(), self.arglist.max_grad_norm)
                    self.optimizers_cr[agent_idx].step()

                # update the tar par

                update_trainers(self.rnn_cur, self.rnn_tar, self.arglist.tao)
                update_trainers(self.actors_cur, self.actors_tar, self.arglist.tao)
                update_trainers(self.critics_cur, self.critics_tar, self.arglist.tao)



class behavior_trainers():
    def __init__(self, env, num_adversaries, obs_shape_n, action_shape_n, arglist):
        """
        init the trainers or load the old model
        """
        self.arglist = arglist
        Transition = namedtuple('Transition',
                                ('state', 'c', 'action', 'next_state', 'nc', 'reward', 'done'))
        self.memory = ReplayBuffer(arglist.memory_size, arglist.kx, transition=Transition)
        self.attention_cur = [None for _ in range(env.n)]
        self.actors_cur = [None for _ in range(env.n)]
        self.critics_cur = [None for _ in range(env.n)]

        self.attention_tar = [None for _ in range(env.n)]
        self.actors_tar = [None for _ in range(env.n)]
        self.critics_tar = [None for _ in range(env.n)]

        self.optimizers_at = [None for _ in range(env.n)]
        self.optimizers_cr = [None for _ in range(env.n)]
        self.optimizers_ac = [None for _ in range(env.n)]
        self.input_size_global = sum(obs_shape_n) + sum(action_shape_n)
        self.update_cnt = 0
        self.n = env.n

        # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
        for i in range(env.n):
            self.attention_cur[i] = SelfAttention(obs_shape_n[i]*2, arglist).to(arglist.device)
            self.actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            self.critics_cur[i] = openai_critic(obs_shape_n[i], sum(action_shape_n), arglist).to(arglist.device)

            self.attention_tar[i] = SelfAttention(obs_shape_n[i]*2, arglist).to(arglist.device)
            self.actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            self.critics_tar[i] = openai_critic(obs_shape_n[i], sum(action_shape_n), arglist).to(arglist.device)

            self.optimizers_at[i] = optim.Adam(self.attention_cur[i].parameters(), arglist.lr_c)
            self.optimizers_ac[i] = optim.Adam(self.actors_cur[i].parameters(), arglist.lr_a)
            self.optimizers_cr[i] = optim.Adam(self.critics_cur[i].parameters(), arglist.lr_c)

        update_trainers(self.attention_cur, self.attention_tar, 1.0)
        update_trainers(self.actors_cur, self.actors_tar, 1.0) # update the target par using the cur
        update_trainers(self.critics_cur, self.critics_tar, 1.0) # update the target par using the cur

    def step(self, obs_n, c):

        torch_obs_n = [torch.from_numpy(obs).to(self.arglist.device, torch.float) for obs in obs_n]
        torch_c = torch.from_numpy(c).to(self.arglist.device, torch.float)

        s_n = [att((obs.unsqueeze(0).detach(),torch_c.unsqueeze(0).detach())) \
                    for att, obs in zip(self.attention_cur, torch_obs_n)]


        action_n = [agent(s).squeeze(0).detach().cpu().numpy() \
                    for agent, s in zip(self.actors_cur, s_n)]
        return action_n


    def train(self, game_step):
        """
            use this func to make the "main" func clean
            par:
            |input: the data for training
            |output: the data for next update
            """
        # update all trainers, if not in display or benchmark mode
        if game_step > self.arglist.learning_start_step and \
                (game_step - self.arglist.learning_start_step) % self.arglist.learning_fre == 0:
            if self.update_cnt == 0: print('\r=start training ...' + ' ' * 100)
            # update the target par using the cur
            self.update_cnt += 1
            if self.arglist.if_weighted_sample:
                transitions = self.memory.weighted_sample(self.arglist.batch_size)
            else:
                transitions = self.memory.sample(self.arglist.batch_size)
            batch = self.memory.transition(*zip(*transitions))  # 转换成为一批次
            obs = torch.tensor(batch.state, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)
            c = torch.tensor(batch.c, dtype=torch.float, device=self.arglist.device)
            actions = torch.tensor(batch.action, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)
            next_obs = torch.tensor(batch.next_state, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)
            nc = torch.tensor(batch.nc, dtype=torch.float, device=self.arglist.device)
            rews = torch.tensor(batch.reward, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)
            dones = torch.tensor(batch.reward, dtype=torch.float, device=self.arglist.device).transpose(0, 1).split(1, 0)
            #action 3 ,1,1256,5
            all_actions = torch.cat(actions,dim=-1).squeeze(0)
            all_n_act = []
            for agent_idx in range(self.n):
                obs_n_n = next_obs[agent_idx].squeeze(0)
                s_tar = self.attention_tar[agent_idx]((obs_n_n, nc)).detach()
                act_cur = self.actors_tar[agent_idx](s_tar).detach()
                all_n_act.append(act_cur)
            all_n_act = torch.cat(all_n_act,dim=-1).detach()
            #print(all_actions.size())
            # update every agent in different memory batch
            for agent_idx in range(self.n):
                rew = rews[agent_idx].squeeze(0)  # set the rew to gpu
                done_n = dones[agent_idx].squeeze(0)  # set the rew to gpu
                action_cur_o = actions[agent_idx].squeeze(0)
                obs_n_o = obs[agent_idx].squeeze(0)
                obs_n_n = next_obs[agent_idx].squeeze(0)

                s = self.attention_cur[agent_idx]((obs_n_o, c))
                s_tar = self.attention_tar[agent_idx]((obs_n_n, nc)).detach()
                #action_tar = [self.actors_tar[agent_idx](s_tar).detach()]

                q = self.critics_cur[agent_idx](s, all_actions).reshape(-1)  # q
                q_ = self.critics_tar[agent_idx](s_tar, all_n_act).detach().reshape(-1)  # q_
                tar_value = q_ * self.arglist.gamma * done_n + rew  # q_*gamma*done + reward
                loss_function = torch.nn.MSELoss()
                loss_c = loss_function(q, tar_value)  # bellman equation
                self.optimizers_cr[agent_idx].zero_grad()
                self.optimizers_at[agent_idx].zero_grad()
                loss_c.backward()
                nn.utils.clip_grad_norm_(self.critics_cur[agent_idx].parameters(), self.arglist.max_grad_norm)
                self.optimizers_cr[agent_idx].step()
                self.optimizers_at[agent_idx].step()

                # --use the data to update the ACTOR
                # There is no need to cal other agent's action
                all_actions_cur = all_actions.clone()
                s = self.attention_cur[agent_idx]((obs_n_o, c))
                model_out, policy_c_new = self.actors_cur[agent_idx](s, model_original_out=True)
                action_size = policy_c_new.size()[-1]
                all_actions_cur[:,action_size*agent_idx:action_size*(agent_idx+1)]=policy_c_new

                #action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
                # update the aciton of this agent
                loss_pse = torch.mean(torch.pow(model_out, 2))
                loss_a = torch.mul(-1, torch.mean(self.critics_cur[agent_idx](s, all_actions_cur)))
                """
                for name, param in self.actors_cur[agent_idx].named_parameters():
                    print(name, param)
                input()
                """
                self.optimizers_ac[agent_idx].zero_grad()
                self.optimizers_at[agent_idx].zero_grad()
                #loss_a.backward()
                (1e-3 * loss_pse + loss_a).backward()
                nn.utils.clip_grad_norm_(self.actors_cur[agent_idx].parameters(), self.arglist.max_grad_norm)
                self.optimizers_ac[agent_idx].step()
                self.optimizers_at[agent_idx].step()
                """
                for name, param in self.actors_cur[agent_idx].named_parameters():
                    print(name,param)
                input()
                """
            # update the tar par

            update_trainers(self.attention_cur, self.attention_tar, self.arglist.tao)
            update_trainers(self.actors_cur, self.actors_tar, self.arglist.tao)
            update_trainers(self.critics_cur, self.critics_tar, self.arglist.tao)
