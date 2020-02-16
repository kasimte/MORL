# What is the difference between run e3c and run e3c double?

## multi-objective super mario bros
## modified by Runzhe Yang on Dec. 18, 2018

import gym
import os
import random
import argparse
from itertools import chain

import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
import time
from datetime import datetime

from model import *

import torch.optim as optim
from torch.multiprocessing import Pipe, Process

from collections import deque

from tensorboardX import SummaryWriter
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from env import MoMarioEnv
from agent import EnveDoubleMoActorAgent

parser = argparse.ArgumentParser(description='MORL')

# set envrioment id and directory
parser.add_argument('--env-id', default='SuperMarioBros-v2', metavar='ENVID',
                    help='Super Mario Bros Game 1-2 (Skip Frame) v0-v3')
parser.add_argument('--name', default='e3c', metavar='name',
                    help='specify the model name')
parser.add_argument('--logdir', default='logs/', metavar='LOG',
                    help='path for recording training informtion')
parser.add_argument('--prev-model', default='SuperMarioBros-v3_2018-11-24.model', metavar='PREV',
                    help='specify the full name of the previous model')

# running configuration
parser.add_argument('--use-cuda', action='store_true',
                    help='use cuda (default FALSE)')

# GAE: Generalized Advantage Estimation
# See
# http://www.breloff.com/DeepRL-OnlineGAE/
# for a summary
parser.add_argument('--use-gae', action='store_true',
                    help='use gae (default FALSE)')
parser.add_argument('--life-done', action='store_true',
                    help='terminate when die')
parser.add_argument('--single-stage', action='store_true',
                    help='only train on one stage ')
parser.add_argument('--load-model', action='store_true',
                    help='load previous model (default FALSE)')
parser.add_argument('--training', action='store_true',
                    help='run for training (default FALSE)')
parser.add_argument('--render', action='store_true',
                    help='render the game (default FALSE)')
parser.add_argument('--standardization', action='store_true',
                    help='load previous model (default FALSE)')
parser.add_argument('--num-worker', type=int, default=16, metavar='NWORKER',
                    help='number of wokers (defualt 16)')

# hyperparameters
parser.add_argument('--lam', type=float, default=0.95, metavar='LAM',
                    help='lambda for gae (default 0.95)')
parser.add_argument('--beta', type=float, default=0.95, metavar='LAM',
                    help='beta for balancing l1 and l2 loss')
parser.add_argument('--T', type=float, default=10, metavar='TEMP',
                    help='softmax with tempreture to encorage exploration')
parser.add_argument('--num-step', type=int, default=5, metavar='NSTEP',
                    help='number of gae steps (default 5)')
parser.add_argument('--max-step', type=int, default=1.15e8, metavar='MSTEP',
                    help='max number of steps for learning rate scheduling (default 1.15e8)')
parser.add_argument('--learning-rate', type=float, default=2.5e-4, metavar='LR',
                    help='initial learning rate (default 2.5e-4)')
parser.add_argument('--lr-schedule', action='store_true',
                    help='enable learning rate scheduling')
parser.add_argument('--enve-start', type=int, default=1e5, metavar='ESTART',
                    help='minimum number of naive traning before envelope')
parser.add_argument('--update-target-critic', type=int, default=1e4, metavar='UTC',
                    help='the number of steps to update target critic')
parser.add_argument('--entropy-coef', type=float, default=0.02, metavar='ENTROPY',
                    help='entropy coefficient for regurization (default 0.2)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='GAMMA',
                    help='gamma for discounted rewards (default 0.99)')
parser.add_argument('--clip-grad-norm', type=float, default=0.5, metavar='CLIP',
                    help='gradient norm clipping (default 0.5)')
parser.add_argument('--reward-scale', type=float, default=1.0, metavar='RS',
                    help='reward scaling (default 1.0)')
parser.add_argument('--sample-size', type=int, default=8, metavar='SS',
                    help='number of preference samples for updating')

# what is this doing? do we need this too?
# what is done[t]? no df
def make_train_data(args, reward, done, value, next_value, reward_size):
    '''
    :param args: arguments
    :param reward: total MORL reward

    :param done: a vector indicating whether each transition is terminal 
p
    E.g., [0, 1, 1, 1, 0....]
    0 --> means not terminal
    1 --> means terminal

    done[t] --> 0 or 1, indicating whether this time step is terminal

    :param value: value vector
    :param next_value: next value vector
    :param reward_size: a number
    :return: discounted_return target to give to envelope_operator
    '''

    # Initialize a discounted_return numpy array of size
    # number of steps, reward_size
    discounted_return = np.empty([args.num_step, reward_size])

    # Discounted Return
    if args.use_gae:
        # Generalized Advantage Estimator
        gae = np.zeros(reward_size)
        for t in range(args.num_step - 1, -1, -1):
            # reward at time step t +
            # delta is the difference between what the model predicts and what the target model is predicting
            delta = reward[t] + args.gamma * next_value[t] * (1 - done[t]) - value[t]
            gae = delta + args.gamma * args.lam * (1 - done[t]) * gae

            discounted_return[t] = gae + value[t]

    else:
        # Start with the last next value.
        running_add = next_value[-1]
        # Iterate backwards in time over the number of steps.
        for t in range(args.num_step - 1, -1, -1):
            # The running addition at each time step becomes the
            # reward at that time step + a discounted figure of the
            # running_add.
            running_add = reward[t] + args.gamma * running_add * (1 - done[t])
            # Update the discounted return.
            discounted_return[t] = running_add


    return discounted_return


# Q. What is this doing? Do we need this too?
# what is this doing? do we need this too?
# return the Q which optimize "????"
# adv is used to calculated Actor_Loss in the agent/train
def envelope_operator(args, preference, target, value, reward_size, g_step):
    '''
    :param args:
    :param preference: update_w = generate_w()
    :param target: target = make_train_data(), is this the real Q?
    :param value: value, next_value, policy = agent.forward_transition(
                total_state, total_next_state, total_update_w)
    :param reward_size: 5
    :param g_step: global_step = args.num_worker * args.num_step = 16 * 5
    :return: 'total_target, total_adv', and input them to the train()
    '''
    
    # Q. what is u1... utility??? This might be a question for Runzhe.
    # [w1, w1, w1, w1, w1, w1,    w2, w2, w2, w2, w2, w2...]
    # [s1, s2, s3, u1, u2, u3,    s1, s2, s3, u1, u2, u3...]

    # weak envelope calculation

    # What is ofs?
    ofs = args.num_worker * args.num_step

    # What does concatenate do?
    target = np.concatenate(target).reshape(-1, reward_size)

    # If our step count is past where we have set it to start, then
    # perform more edits on the target.
    if g_step > args.enve_start:
        
        prod = np.inner(target, preference)

        # envelope_mask?
        envemask = prod.transpose().reshape(args.sample_size, -1, ofs).argmax(axis=1)

        envemask = envemask.reshape(-1) * ofs + np.array(list(range(ofs))*args.sample_size)
        target = target[envemask]

    # For Actor
    # Q = state Value function V(s) + advantage value A(s, a)
    # adv = Q - state Value function V(s)
    # value: Critic value given states
    adv = target - value

    return target, adv


def generate_w(num_prefence, reward_size, fixed_w=None):
    '''
    Generates weight preferences, sampling from a distribution.
    Later called as
    
    explore_w = generate_w(...

    implying that these are exploratory weights.
    '''
    if fixed_w is not None:
        w = np.random.randn(num_prefence-1, reward_size)
        # normalize as a simplex
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence-1, 1)
        return np.concatenate(([fixed_w], w))
    else:
        w = np.random.randn(num_prefence-1, reward_size)
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w

def renew_w(preferences, dim):
    w = np.random.randn(reward_size)
    w = np.abs(w) / np.linalg.norm(w, ord=1, axis=0)
    preferences[dim] = w
    return preferences


if __name__ == '__main__':

    # Parse arguments from configuration.
    args = parser.parse_args()

    # Get environment information.
    env = JoypadSpace(gym_super_mario_bros.make(args.env_id), SIMPLE_MOVEMENT)
    input_size = env.observation_space.shape
    output_size = env.action_space.n

    # Vectorized reward size, representing multiple objectives:
    # [x-position, time, life, coins, enemy]
    # as specified in the paper.
    reward_size = 5

    # Close the environment.
    env.close()

    # Setup
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    # tag is "test" or "training" depending on args.training
    tag = ["test", "train"][int(args.training)]
    log_dir = os.path.join(args.logdir, '{}_{}_{}_{}'.format(
        args.env_id, args.name, current_time, tag))
    writer = SummaryWriter(log_dir)

    model_path = 'saved/{}_{}_{}.model'.format(args.env_id, args.name, current_time)
    load_model_path = 'saved/{}'.format(args.prev_model)

    # EnvelopeDoubleMORLActorAgent
    #
    # - Envelope MORL algorithm
    # - Double model for stabilization
    # - Actor in Actor Critic
    agent = EnveDoubleMoActorAgent(
        args,
        input_size,
        output_size,
        reward_size)

    # Load the model, if applicable
    if args.load_model:
        if args.use_cuda:
            agent.model.load_state_dict(torch.load(load_model_path))
            agent.model_ = copy.deepcopy(agent.model)
        else:
            agent.model.load_state_dict(
                torch.load(
                    load_model_path,
                    map_location='cpu'))
            agent.model_ = copy.deepcopy(agent.model)

    # If testing, run evaluation.
    # eval just prints out what the model looks like
    if not args.training:
        agent.model.eval()
        agent.model_.eval()

    # Initialize array of multiple works for multiprocessing.
    #
    # Each work is a MoMarioEnv (MORL Mario Environment), as seen in
    # the loop below. Each work will be spawned as a child process,
    # passing data up towards its parent through a pipe.
    works = []

    # Parent Connections
    parent_conns = []
    # Child Connections
    child_conns = []

    # Iterate over the number of workers.
    for idx in range(args.num_worker):

        # Pipe is from torch.multiprocessing
        #
        # https://docs.python.org/3.5/library/multiprocessing.html#multiprocessing.Pipe

        # When using multiple processes, one generally uses message
        # passing for communication between processes and avoids
        # having to use any synchronization primitives like locks.

        # For passing messages one can use Pipe() (for a connection
        # between two processes) or a queue (which allows multiple
        # producers and consumers).
        
        parent_conn, child_conn = Pipe()
        work = MoMarioEnv(args, idx, child_conn)
        work.start()
        works.append(work)
        parent_conns.append(parent_conn)
        child_conns.append(child_conn)

    # Initialize states, being the an array of states (one state from each worker).
    # Where does 4, 84, 84 come from here?
    # Presumably that is the state shape for each worker.
    states = np.zeros([args.num_worker, 4, 84, 84])

    sample_episode = 0
    # Sample Reward All
    sample_rall = 0
    # Sample MORL Reward All
    sample_morall = 0
    sample_step = 0
    sample_env_idx = 0
    global_step = 0
    recent_prob = deque(maxlen=10)

    # Fixed Weights
    fixed_w = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    # 
    # fixed_w = np.array([0.00, 0.00, 0.00, 1.00, 0.00])
    # fixed_w = np.array([0.00, 0.00, 0.00, 0.00, 1.00])
    # fixed_w = np.array([1.00, 0.00, 0.00, 0.00, 0.00])
    # fixed_w = np.array([0.00, 1.00, 0.00, 0.00, 0.00])
    # fixed_w = np.array([0.00, 0.00, 1.00, 0.00, 0.00])
    explore_w = generate_w(args.num_worker, reward_size, fixed_w)

    # Not sure why this loop is necessary.
    while True:
        total_state, total_reward, total_done, total_next_state, total_action, total_moreward = [], [], [], [], [], []
        global_step += (args.num_worker * args.num_step)

        # num_step - number of steps, defaults to 5
        # This is done for GAE.
        for _ in range(args.num_step):

            # Not sure why we delay here during testing, but okay.
            if not args.training:
                time.sleep(0.05)

            # The agent returns the actions based on states. I guess
            # the agent is returning actions as a vector, indicating
            # the agent is setup to handle multiple actors at once.

            # states: [ worker 1 state, worker 2 state, ... worker n state ]
            # explore_w: [ weights 1, weights 2, .. weights n ]
            # returns:
            # actions [ action 1, action 2, ... action n ]
            actions = agent.get_action(states, explore_w)

            # Each worker represents an agent, so zip the
            # multiprocessing connection with the access, and send
            # down each action.
            for parent_conn, action in zip(parent_conns, actions):
                parent_conn.send(action)

            # Initialize arrays for data on the flip side.
            #
            # rewards and dones are for plotting, while real_dones and
            # morewards are for the real environment.
            next_states, rewards, dones, real_dones, morewards, scores = [], [], [], [], [], []
            cnt = 0

            # Iterate across workers
            for parent_conn in parent_conns:
                # state, reward, done, real dones, morl reward, scores
                s, r, d, rd, mor, sc = parent_conn.recv()
                next_states.append(s)
                rewards.append(fixed_w.dot(mor))
                dones.append(d)
                real_dones.append(rd)
                morewards.append(mor)
                scores.append(sc)
                # Resample if done
                #
                # (Kasim) What is interesting here is weights are
                # renewed if it is done for any connection.
                #
                if cnt > 0 and d:
                    explore_w = renew_w(explore_w, cnt)
                cnt += 1

            # Stack data appropriately before appending to totals.
            next_states = np.stack(next_states)
            # reward_scale is a hyperparameter that defaults to 1
            rewards = np.hstack(rewards) * args.reward_scale
            dones = np.hstack(dones)
            real_dones = np.hstack(real_dones)
            morewards = np.stack(morewards) * args.reward_scale

            total_state.append(states)
            total_next_state.append(next_states)
            total_reward.append(rewards)
            total_done.append(dones)
            total_action.append(actions)
            total_moreward.append(morewards)

            # Why is it 4 here?
            states = next_states[:, :, :, :]

            # Sampling, presumably for plotting.
            sample_rall += rewards[sample_env_idx]
            sample_morall = sample_morall + morewards[sample_env_idx]
            sample_step += 1
            if real_dones[sample_env_idx]:
                sample_episode += 1
                agent.anneal()
                writer.add_scalar('data/reward', sample_rall, sample_episode)
                writer.add_scalar('data/step', sample_step, sample_episode)
                writer.add_scalar('data/score', scores[sample_env_idx], sample_episode)
                writer.add_scalar('data/x_pos_reward', sample_morall[0], sample_episode)
                writer.add_scalar('data/time_penalty', sample_morall[1], sample_episode)
                writer.add_scalar('data/death_penalty', sample_morall[2], sample_episode)
                writer.add_scalar('data/coin_reward', sample_morall[3], sample_episode)
                writer.add_scalar('data/enemy_reward', sample_morall[4], sample_episode)
                writer.add_scalar('data/tempreture', agent.T, sample_episode)
                sample_rall = 0
                sample_step = 0
                sample_morall = 0

        # If training, learn the model
        if args.training:
            # reward size is 5,
            # [w1, w1, w1, w1, w1, w1, w2, w2, w2, w2, w2, w2...]
            # [s1, s2, s3, u1, u2, u3, s1, s2, s3, u1, u2, u3...]
            # expand w batch
            # First generate some random weights
            update_w = generate_w(args.sample_size, reward_size, fixed_w)
            # Then expand that to total by repeating it, resulting in the data structure above.
            total_update_w = update_w.repeat(args.num_step*args.num_worker, axis=0)
            
            # expand state batch
            # WRONG!!! total_state = total_state * args.sample_size
            total_state = np.stack(total_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_state = np.tile(total_state, (args.sample_size, 1, 1, 1))
            # expand next_state batch
            # WRONG!!! total_next_state = total_next_state * args.sample_size
            total_next_state = np.stack(total_next_state).transpose(
                [1, 0, 2, 3, 4]).reshape([-1, 4, 84, 84])
            total_next_state = np.tile(total_next_state, (args.sample_size, 1, 1, 1))

            # calculate utility from reward vectors
            # what is utility? no use
            total_moreward = np.array(total_moreward).transpose([1, 0, 2]).reshape([-1, reward_size])
            total_moreward = np.tile(total_moreward, (args.sample_size, 1))

            # Question: Where is this total_utility used?
            total_utility = np.sum(total_moreward * total_update_w, axis=-1).reshape([-1])
            
            # expand action batch
            total_action = np.stack(total_action).transpose().reshape([-1])
            total_action = np.tile(total_action, args.sample_size)
            # expand done batch
            total_done = np.stack(total_done).transpose().reshape([-1])
            total_done = np.tile(total_done, args.sample_size)

            
            # forward_transition calculates
            # value, next_value, and policy, based on batches of state, next state, and update weights
            value, next_value, policy = agent.forward_transition(total_state, total_next_state, total_update_w)

            # logging utput to see how convergent it is.
            policy = policy.detach()
            m = F.softmax(policy, dim=-1)
            recent_prob.append(m.max(1)[0].mean().cpu().numpy())
            writer.add_scalar(
                'data/max_prob',
                np.mean(recent_prob),
                sample_episode)

            total_adv = []
            total_target =[]

            # this loops makes a batch of target and advantage
            # training data based on the sample size
            for idw in range(args.sample_size):
                ofs = args.num_worker * args.num_step
                for idx in range(args.num_worker):
                    target = make_train_data(args,
                                  total_moreward[idx*args.num_step+idw*ofs : (idx+1)*args.num_step+idw*ofs],
                                  total_done[idx*args.num_step+idw*ofs: (idx+1)*args.num_step+idw*ofs],
                                  value[idx*args.num_step+idw*ofs : (idx+1)*args.num_step+idw*ofs],
                                  next_value[idx*args.num_step+idw*ofs : (idx+1)*args.num_step+idw*ofs],
                                  reward_size)
                    total_target.append(target)

            print(total_target, "--------before envelope----------")

            # the envelope operator then calculates batches of target
            # and advantage data based on the above
            total_target, total_adv = envelope_operator(args, update_w, total_target, value, reward_size, global_step)
            print(total_target, "--------after envelope----------")

            # finally, we can train the model with batches of the
            # following
            agent.train_model(
                total_state,
                total_next_state,
                total_update_w,
                total_target,
                total_action,
                total_adv)

            # adjust learning rate
            if args.lr_schedule:
                new_learing_rate = args.learning_rate - \
                    (global_step / args.max_step) * args.learning_rate
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = new_learing_rate
                    writer.add_scalar(
                        'data/lr', new_learing_rate, sample_episode)

            # saving
            if global_step % (args.num_worker * args.num_step * 100) == 0:
                torch.save(agent.model.state_dict(), model_path)

            # syncing
            if global_step % args.update_target_critic == 0:
                agent.sync()
