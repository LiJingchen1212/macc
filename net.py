# Time: 2019-11-05
# Author: Zachary
# Name: MADDPG_torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()

    def act(self, input):
        policy, value = self.forward(input)  # flow the input through the nn
        return policy, value


class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)

    def forward(self, input):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        policy = self.tanh(self.linear_a(x))
        return policy


class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_1)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1 * 2, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_o = self.LReLU(self.linear_o_c1(obs_input))
        x_a = self.LReLU(self.linear_a_c1(action_input))
        x_cat = torch.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


class openai_critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        print("obs",obs_shape_n)
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n + obs_shape_n, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


class openai_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(openai_actor, self).__init__()
        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai)
        self.linear_a2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_a = nn.Linear(args.num_units_openai, action_size)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        """
        if model_original_out == True:   return model_out, F.softmax(model_out)  # for model_out criterion
        return policy
        """
        if model_original_out == True:   return model_out, policy  # for model_out criterion
        return policy

class collaboration_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(collaboration_actor, self).__init__()
        self.tanh = nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai)
        self.linear_a2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_a = nn.Linear(args.num_units_openai, action_size)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        return model_out


class noisy_RNN(nn.Module):
    def __init__(self, num_inputs, args):
        super(noisy_RNN, self).__init__()
        self.arglist = args
        self.gru = nn.GRU(num_inputs, self.arglist.num_units_openai, num_layers=1, bias=False)
        self.train()
        self.init_gru_state()

    def init_gru_state(self):
        self.hs = torch.zeros([1, self.arglist.ta, self.arglist.num_units_openai]).to(self.arglist.device, torch.float)
        self.h = torch.zeros([1, 1, self.arglist.num_units_openai]).to(self.arglist.device, torch.float)

    def forward(self, ips, ifbatch=False):
        ips = ips.unsqueeze(0)
        if (ifbatch):
            out, self.hs = self.gru(ips, self.hs)
        else:
            out, self.h = self.gru(ips, self.h)
        return out.squeeze(0)


class SelfAttention(nn.Module):
    def __init__(self, input_dim, arg):
        super(SelfAttention, self).__init__()

        # 定义线性变换函数
        self.linear_1 = nn.Linear(input_dim, 2, bias=False)
        self.linear_2 = nn.Linear(input_dim, arg.num_units_openai, bias=False)
        self.linear_3 = nn.Linear(input_dim, arg.num_units_openai, bias=False)
        self._norm_fact = 1 / math.sqrt(arg.num_units_openai)

    def forward(self, x):
        # 这里将x视为词向量长度为1的序列，下列相乘顺序与原self-attention有出入仅为方便代码编写
        a,b = x
        #return a
        x = torch.cat((a,b),dim=-1)
        w = torch.softmax(self.linear_1(x),dim=-1)
        out = w[0,0]*a+w[0,1]*b
        return out
        """
        x = x.unsqueeze(1)
        # x: batch, n, dim_q

        #q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v

        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q.transpose(1, 2), k) * self._norm_fact  # batch, n, n
        print(dist)
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(v, dist)
        return att.squeeze(1)
        """
