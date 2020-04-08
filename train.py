import os
import numpy as np
import torch
import torch.nn.functional as F
import gym
import datetime
import torch.multiprocessing as mp

from torch.utils.tensorboard import SummaryWriter
from actor_critic import ActorCritic

# Hyperparameters
NUM_AGENTS      = 16
BATCH_SIZE      = 32
LEARNING_RATE   = 5e-4

GAMMA           = 0.99

def test_model(model):
    env = gym.make('CartPole-v1')

    score = 0.

    s = env.reset()
    while True:
        a = model.action(s)
        s, r, d, _ = env.step(a)
        score += r
        if d:
            break

    return score

def compute_adv(rewards, values, dones, value_n):
    returns = np.append(np.zeros_like(rewards), [value_n], axis=-1)
    for t in reversed(range(len(rewards))):
        returns[t] = rewards[t] + GAMMA * returns[t + 1] * (1 - dones[t])
        returns = returns[:-1]
    adv = returns - values
    return adv, returns

def get_exp(exp_queues, state_dim):
    batch_size = NUM_AGENTS * BATCH_SIZE
    states = np.empty((batch_size, state_dim))
    actions, rewards, dones, advs, returns = [np.empty((batch_size,)) for _ in range(5)]
    staten = []

    for i in range(NUM_AGENTS):
        s, a, r, d, ad, g = exp_queues[i].get()
        idx_s = i * BATCH_SIZE
        idx_e = i * BATCH_SIZE + BATCH_SIZE
        states[idx_s:idx_e, :] = s
        actions[idx_s:idx_e] = a
        rewards[idx_s:idx_e] = r
        dones[idx_s:idx_e] = d
        advs[idx_s:idx_e] = ad
        returns[idx_s:idx_e] = g

    return states, actions, rewards, dones, advs, returns

def central_agent(exp_queues, model_queues):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    # Create tensorboard
    s_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = '../tensorboard/' + 'CartPole-A2C_' + s_time
    train_summary_writer = SummaryWriter(dir_path)

    env = gym.make('CartPole-v1')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, lr=LEARNING_RATE)

    num_updates = 0
    while True:
        ############################################################################
        # Synchronize model with worker agents                                     #
        ############################################################################
        state_dict = model.state_dict()
        for i in range(NUM_AGENTS):
            model_queues[i].put(state_dict)

        ############################################################################
        # Recieve trajectories                                                     #
        ############################################################################
        states, actions, rewards, dones, advs, returns = get_exp(exp_queues, env.observation_space.shape[0])

        ############################################################################
        # Train                                                                    #
        ############################################################################
        # convert from numpy to tensors
        states = torch.tensor(states.copy(), device=device).float()
        actions = torch.tensor(actions.copy(), device=device).long()[:, None]
        advs = torch.tensor(advs.copy(), device=device).float()[:, None]
        returns = torch.tensor(returns.copy(), device=device).float()[:, None]

        optimizer = model.optimizer
        optimizer.zero_grad()
        ppreds, vpreds = model(states)
        probs = ppreds.gather(1, actions)
        loss_policy = (-torch.log(probs) * advs).mean()
        loss_value = torch.nn.MSELoss()(vpreds, returns)
        loss = loss_policy + loss_value
        loss.backward()
        optimizer.step()

        num_updates += 1

        ############################################################################
        # Record to tensorboard                                                    #
        ############################################################################
        dist = torch.distributions.Categorical(probs=ppreds.detach())
        entropy = dist.entropy().mean().numpy()

        train_summary_writer.add_scalar('Loss/Total', loss, num_updates)
        train_summary_writer.add_scalar('Loss/Actor', loss_policy, num_updates)
        train_summary_writer.add_scalar('Loss/Critic', loss_value, num_updates)
        train_summary_writer.add_scalar('Loss/Entropy', entropy, num_updates)
        train_summary_writer.add_scalar('test_epi_rewards', test_model(model), num_updates)

def agent(i, exp_queue, model_queue):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    env = gym.make('CartPole-v1')
    model = ActorCritic(env.observation_space.shape[0], env.action_space.n, lr=LEARNING_RATE)

    states = np.empty((BATCH_SIZE, env.observation_space.shape[0]))
    actions = np.empty((BATCH_SIZE,), dtype=np.int32)
    rewards = np.empty((BATCH_SIZE,))
    dones = np.empty((BATCH_SIZE,))

    num_updates = 0

    sn = env.reset()
    while True:
        # Synchronize model with central agent
        model_path = model_queue.get()
        model.load_state_dict(model_path)

        # Experience
        for t in range(BATCH_SIZE):
            states[t] = sn.copy()
            actions[t], _, _ = model.action_sample(states[t])
            sn, rewards[t], dones[t], _ = env.step(actions[t])

            if dones[t]:
                sn = env.reset()

        # compute advantages
        values = model.values(states)
        valuen = model.value(sn)

        advs, returns = compute_adv(rewards, values, dones, valuen)

        exp_queue.put((
            states.copy(),
            actions.copy(),
            rewards.copy(),
            dones.copy(),
            advs.copy(),
            returns.copy()
        ))

if __name__ == '__main__':
    ############################################################################
    # Create queues for communication                                          #
    ############################################################################
    exp_queues = []
    model_queues = []
    for i in range(NUM_AGENTS):
        exp_queues.append(mp.Queue(1))
        model_queues.append(mp.Queue(1))

    ############################################################################
    # Create processes                                                         #
    ############################################################################
    # central agent
    central = mp.Process(target=central_agent, args=(exp_queues, model_queues))
    central.start()

    # worker agent
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(target=agent, args=(i, exp_queues[i], model_queues[i])))

    for i in range(NUM_AGENTS):
        agents[i].start()

    ############################################################################
    # Join processes                                                           #
    ############################################################################
    central.join()

    for i in range(NUM_AGENTS):
        agents[i].join()