from __future__ import absolute_import, division, print_function
import random
import torch
import copy
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from collections import namedtuple
from collections import deque

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


class MetaAgent(object):
    '''
    (1) act: how to sample an action to examine the learning
        outcomes or explore the environment;
    (2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
    (3) learn: how the agent learns from the observations via
        explicitor implicit inference, how to optimize the policy
        model.
    '''

    # model: likely something from the `./models` folder:
    # - linear.py
    # - linear_ext.py
    # - onelayer.py
    def __init__(self, model, args, is_train=False):
        #
        # more on copying:
        # https://docs.python.org/3.5/library/copy.html#copy.deepcopy
        #
        # model_ is a shallow copy of the model
        self.model_ = model
        # self.model is a deep copy of the model
        self.model = copy.deepcopy(model)
        self.is_train = is_train
        # what is gamma here? not sure. i forget
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_delta = (args.epsilon - 0.05) / args.episode_num

        self.mem_size = args.mem_size
        self.batch_size = args.batch_size
        self.weight_num = args.weight_num

        # trans_mem is likely transition memory
        # more on deque:
        # https://docs.python.org/2/library/collections.html#deque-objects
        self.trans_mem = deque()
        # trans: a tuple that includes the transition
        # - s: state
        # - a: action
        # - s_: next state
        # - r: reward
        # - d: ? not sure what d is
        self.trans = namedtuple('trans', ['s', 'a', 's_', 'r', 'd'])
        self.priority_mem = deque()

        # the optimizer can be either adam or rmsprop
        if args.optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model_.parameters(), lr=args.lr)
        elif args.optimizer == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model_.parameters(), lr=args.lr)

        # weights?
        self.w_kept = None
        self.update_count = 0
        self.update_freq = args.update_freq

        if self.is_train:
            self.model_.train()
        if use_cuda:
            self.model.cuda()
            self.model_.cuda()

    def act(self, state, preference=None):
        """How to sample an action to examine the learning
        outcomes or explore the environment
        """

        # random pick a preference if it is not specified
        if preference is None:
            # w_kept here is presumably the kept weights
            if self.w_kept is None:
                # if we don't have kept wegihts, randomize
                self.w_kept = torch.randn(self.model_.reward_size)
                # and then normalize (i think)
                self.w_kept = (torch.abs(self.w_kept) / \
                               torch.norm(self.w_kept, p=1)).type(FloatTensor)
            preference = self.w_kept

        # pull out the state into a format we want
        state = torch.from_numpy(state).type(FloatTensor)

        # run our stuff through the model,
        # which takes the state and a preference
        _, Q = self.model_(
            Variable(state.unsqueeze(0), requires_grad=False),
            Variable(preference.unsqueeze(0), requires_grad=False))

        # pull out the action based on the Q values
        action = Q.max(1)[1].data.cpu().numpy()
        # action formatting
        action = int(action[0])

        # if training, then do some more formatting of the action?
        if self.is_train and (len(self.trans_mem) < self.batch_size or \
                              torch.rand(1)[0] < self.epsilon):
            action = np.random.choice(self.model.action_size, 1)[0]
            action = int(action)

        return action

    # what is roi here?
    def memorize(self, state, action, next_state, reward, terminal, roi=False):
        """(2) memorize: how to store observed observations in order to
        help learing or establishing the empirical model of the
        enviroment;
        """

        # add transition tuple (s, a, s', r, t) to memory
        self.trans_mem.append(self.trans(
            torch.from_numpy(state).type(FloatTensor),  # state
            action,  # action
            torch.from_numpy(next_state).type(FloatTensor),  # next state
            torch.from_numpy(reward).type(FloatTensor),  # reward
            terminal))  # terminal

        # naive model:
        # randomly produce a preference for calculating priority
        #if roi:
        #    preference = self.w_kept
        #else:

        # why is preference random here?
        preference = torch.randn(self.model_.reward_size)
        preference = (torch.abs(preference) / \
                      torch.norm(preference, p=1)).type(FloatTensor)

        # reformat state
        state = torch.from_numpy(state).type(FloatTensor)

        # input state and action into model
        _, q = self.model_(Variable(state.unsqueeze(0), requires_grad=False),
                           Variable(preference.unsqueeze(0), requires_grad=False))
        q = q[0, action].data
        wr = preference.dot(torch.from_numpy(reward).type(FloatTensor))
        if not terminal:
            next_state = torch.from_numpy(next_state).type(FloatTensor)
            hq, _ = self.model_(Variable(next_state.unsqueeze(0), requires_grad=False),
                                Variable(preference.unsqueeze(0), requires_grad=False))
            hq = hq.data[0]
            p = abs(wr + self.gamma * hq - q)
        else:
            self.w_kept = None
            if self.epsilon_decay:
                self.epsilon -= self.epsilon_delta
            p = abs(wr - q)
        p += 1e-5

        #if roi:
        #    p = 1

        # priority is just randomized here
        self.priority_mem.append(
            p
        )
        if len(self.trans_mem) > self.mem_size:
            self.trans_mem.popleft()
            self.priority_mem.popleft()

    def sample(self, pop, pri, k):
        '''Sample a minibatch of transitions from our memory
        pop: transition memory
        pri: priority
        k: batch size
        '''
        # reformat priority as a numpy array, with type float
        pri = np.array(pri).astype(np.float)

        # grab random indicies
        inds = np.random.choice(
            range(len(pop)), k,
            replace=False,
            p=pri / pri.sum()
        )

        # return the sample
        return [pop[i] for i in inds]

    def actmsk(self, num_dim, index):
        mask = ByteTensor(num_dim).zero_()
        mask[index] = 1
        return mask.unsqueeze(0)

    def nontmlinds(self, terminal_batch):
        mask = ByteTensor(terminal_batch)
        inds = torch.arange(0, len(terminal_batch)).type(LongTensor)
        inds = inds[mask.eq(0)]
        return inds

    def learn(self, preference=None):
        # if the transition memory is greater than the batch size
        if len(self.trans_mem) > self.batch_size:

            self.update_count += 1

            # sample a minibatch from our memory
            minibatch = self.sample(self.trans_mem, self.priority_mem, self.batch_size)
            # define a function that takes x, turns it into a list and multiplies it by the weight num
            # weight_num seems to be set to 32 in most configurations
            batchify = lambda x: list(x) * self.weight_num

            state_batch = batchify(map(lambda x: x.s.unsqueeze(0), minibatch))
            action_batch = batchify(map(lambda x: LongTensor([x.a]), minibatch))
            reward_batch = batchify(map(lambda x: x.r.unsqueeze(0), minibatch))
            next_state_batch = batchify(map(lambda x: x.s_.unsqueeze(0), minibatch))
            terminal_batch = batchify(map(lambda x: x.d, minibatch))

            if preference is None:
                w_batch = np.random.randn(self.weight_num, self.model_.reward_size)
                w_batch = np.abs(w_batch) / \
                          np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)
            else:
                w_batch = preference.cpu().numpy()
                w_batch = np.expand_dims(w_batch, axis=0)
                w_batch = np.abs(w_batch) / \
                          np.linalg.norm(w_batch, ord=1, axis=1, keepdims=True)
                w_batch = torch.from_numpy(w_batch.repeat(self.batch_size, axis=0)).type(FloatTensor)
            

            __, Q = self.model_(Variable(torch.cat(state_batch, dim=0)),
                                Variable(w_batch))
            # detach since we don't want gradients to propagate
            # HQ, _    = self.model_(Variable(torch.cat(next_state_batch, dim=0), volatile=True),
            # 					  Variable(w_batch, volatile=True))
            _, DQ = self.model(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                               Variable(w_batch, requires_grad=False))
            _, act = self.model_(Variable(torch.cat(next_state_batch, dim=0), requires_grad=False),
                                 Variable(w_batch, requires_grad=False))[1].max(1)
            HQ = DQ.gather(1, act.unsqueeze(dim=1)).squeeze()

            w_reward_batch = torch.bmm(w_batch.unsqueeze(1),
                                       torch.cat(reward_batch, dim=0).unsqueeze(2)
                                       ).squeeze()

            nontmlmask = self.nontmlinds(terminal_batch)
            with torch.no_grad():
                Tau_Q = Variable(torch.zeros(self.batch_size * self.weight_num).type(FloatTensor))
                Tau_Q[nontmlmask] = self.gamma * HQ[nontmlmask]
                Tau_Q += Variable(w_reward_batch)

            actions = Variable(torch.cat(action_batch, dim=0))

            # Compute Huber loss
            # Huber loss is like squared error, but doesn't punish outliers quite as much.
            # https://en.wikipedia.org/wiki/Huber_loss
            loss = F.smooth_l1_loss(Q.gather(1, actions.unsqueeze(dim=1)), Tau_Q.unsqueeze(dim=1))

            self.optimizer.zero_grad()
            loss.backward()
            for param in self.model_.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()

            if self.update_count % self.update_freq == 0:
                self.model.load_state_dict(self.model_.state_dict())

            return loss.data

        return 0.0

    def reset(self):
        self.w_kept = None
        if self.epsilon_decay:
            self.epsilon -= self.epsilon_delta

    def predict(self, probe):
        return self.model(Variable(FloatTensor([0, 0]).unsqueeze(0), requires_grad=False),
                          Variable(probe.unsqueeze(0), requires_grad=False))

    def save(self, save_path, model_name):
        '''Save the model in pkl format to specified path.'''
        torch.save(self.model, "{}{}.pkl".format(save_path, model_name))
