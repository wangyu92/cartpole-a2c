import torch
import torch.nn.functional as F
import numpy as np

from types import SimpleNamespace


def compute_adv(exp_dict, p):
    exp = SimpleNamespace(**exp_dict)

    returns = np.append(np.zeros_like(exp.rewards), np.zeros((1,1)), axis=0)
    
    for t in reversed(range(len(exp.rewards))):
        returns[t] = exp.rewards[t] + p.GAMMA * returns[t + 1] * (1 - exp.dones[t])
    
    returns = returns[:-1]

    adv = returns - exp.values
    return adv, returns


def merge_exp(exp_queues):
    states, actions, rewards, dones, statesn = [[] for _ in range(5)]
    for queue in exp_queues:
        exp_dict = queue.get()
        exp_dict = SimpleNamespace(**exp_dict)

        states += exp_dict.states.tolist()
        statesn += exp_dict.statesn.tolist()
        actions += exp_dict.actions.tolist()
        rewards += exp_dict.rewards.tolist()
        dones += exp_dict.dones.tolist()

    states, actions, rewards, dones, statesn =\
        map(lambda x: np.array(x), [states, actions, rewards, dones, statesn])

    exp_dict = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'statesn': statesn,
    }

    return exp_dict


def merge_metric(metric_queues):

    episode_rewards = []

    for queue in metric_queues:
        metric_dict = queue.get()
        episode_rewards += metric_dict['episode_rewards']

    metric_dict = {
        'episode_rewards': episode_rewards
    }

    return metric_dict


def train_model(model, exp_dict, num_updates, p):
    losses = []
    losses_p = []
    losses_v = []
    ent_coefs = []
    entropies = []
    
    results = train_step(model, exp_dict, num_updates, p)
    losses.append(results[0])
    losses_p.append(results[1])
    losses_v.append(results[2])
    ent_coefs.append(results[3])
    entropies.append(results[4])

    res_dict = {
        'total': np.mean(losses),
        'actor': np.mean(losses_p),
        'critic': np.mean(losses_v),
        'ent_coefs': np.mean(ent_coefs),
        'entropies': np.mean(entropies)
    }

    return res_dict


def train_step(model, exp_dict, num_updates, p):
    device = model.getdevice()

    exp = SimpleNamespace(**exp_dict)
    states = torch.tensor(exp.states, device=device)
    actions = torch.tensor(exp.actions, device=device)
    advs = torch.tensor(exp.advs, device=device)
    returns = torch.tensor(exp.returns, device=device)

    # back propagation
    optimizer = torch.optim.Adam(model.parameters(), lr=p.LR)
    optimizer.zero_grad()
    ppreds, vpreds = model(states)
    probs = ppreds.gather(1, actions)

    entropy = torch.distributions.Categorical(probs=ppreds).entropy().mean()
    ent_decay = p.ENT_MAX - num_updates * (p.ENT_MAX - p.ENT_MIN) / p.ENT_STEP
    ent_coef = np.clip(ent_decay, p.ENT_MIN, p.ENT_MAX)

    loss_policy = (-torch.log(probs) * advs).mean()
    loss_value = torch.nn.MSELoss()(vpreds, returns)
    loss = loss_policy + loss_value - (ent_coef * entropy)

    loss.backward()
    optimizer.step()

    loss = loss.detach().item()
    loss_policy = loss_policy.detach().item()
    loss_value = loss_value.detach().item()

    return loss, loss_policy, loss_value, ent_coef, entropy.item()







