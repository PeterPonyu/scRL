import numpy as np
import pandas as pd
import pytest
import torch

import scRL
from conftest import make_tree, seed_all


def _ready3d():
    X, labels, Z, genes = make_tree(d=3)
    gres = scRL.grids_from_embedding(X)
    scRL.project_cluster(gres, pd.Series(pd.Categorical(labels)))
    seed_all(0); scRL.align_pseudotime(gres, 'root', boundary=False)
    scRL.project(gres, genes)
    return gres, Z


def _grid(d, scan2d):
    gres, Z = (scan2d, make_tree(d=2)[2]) if d == 2 else _ready3d()
    scRL.d_rewards(gres, starts=['root'], ends=['branchA'])
    return gres, Z


def _sac(gres, Z, episodes, **kw):
    return scRL.trainer('SAC', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=episodes,
                        capacity=2000, **kw)


@pytest.mark.parametrize('d', [2, 3])
def test_sac_trains_and_reports_values(d, scan2d):
    gres, Z = _grid(d, scan2d)
    seed_all(0)
    t = _sac(gres, Z, 300)
    returns, _ = t.train()
    assert len(returns) == 300 and np.isfinite(returns).all()
    t.eval_fate()
    fate = np.asarray(gres.grids['fate_branchA'], dtype=float)
    assert fate.shape == (len(gres.grids['mapped_grids']),) and np.isfinite(fate).all()
    scRL.get_state_value(gres, t, 'v')
    np.testing.assert_array_equal(gres.grids['v'], fate)


def test_sac_runs_the_simulator(scan2d):
    gres, Z = _grid(2, scan2d)
    seed_all(0)
    t = _sac(gres, Z, 50)
    t.train()
    t.train_simulator(20, 8)


@pytest.mark.parametrize('algo', ['ActorCritic', 'DDQN', 'TabularQ', 'SAC'])
def test_get_state_value_equals_eval_fate(algo, scan2d):
    gres, Z = _grid(2, scan2d)
    seed_all(0)
    t = scRL.trainer(algo, gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=50, capacity=2000)
    t.train()
    t.eval_fate()
    scRL.get_state_value(gres, t, 'v')
    np.testing.assert_array_equal(np.asarray(gres.grids['v'], dtype=float),
                                  np.asarray(gres.grids['fate_branchA'], dtype=float))


def test_actor_critic_values_unchanged(scan2d):
    gres, Z = _grid(2, scan2d)
    seed_all(0)
    t = scRL.trainer('ActorCritic', gres, reward_type='d', starts_prob=False, X_latent=Z, num_episodes=50)
    t.train()
    scRL.get_state_value(gres, t, 'v')
    critic = t.agent.critic
    states = torch.tensor(t.env.state_space.mean(axis=1), dtype=torch.float, device=next(critic.parameters()).device)
    np.testing.assert_array_equal(gres.grids['v'], critic(states).detach().cpu().numpy().ravel())


def _agent(monkeypatch):
    from scRL import SACCore
    monkeypatch.setattr(SACCore, 'device', torch.device('cpu'))
    torch.manual_seed(0)
    return SACCore.sac(5, 16, 6, .9, 5e-4, 1e-3, tau=.01)


def test_soft_value_formula(monkeypatch):
    agent = _agent(monkeypatch)
    states = np.random.default_rng(0).normal(size=(7, 5)).astype(np.float32)
    s = torch.tensor(states)
    with torch.no_grad():
        p = agent.actor(s)
        q = torch.min(agent.q1(s), agent.q2(s))
        expected = (p * (q - agent.log_alpha.exp() * torch.log(p + 1e-8))).sum(1)
    np.testing.assert_allclose(agent.value(states).numpy(), expected.numpy(), rtol=1e-6)
    assert agent.v_value(states[0]) == pytest.approx(float(expected[0]), rel=1e-6)


def test_one_update_step(monkeypatch):
    agent = _agent(monkeypatch)
    rng = np.random.default_rng(1)
    batch = {'states': rng.normal(size=(8, 5)), 'actions': rng.integers(6, size=8), 'rewards': rng.normal(size=8),
             'next_states': rng.normal(size=(8, 5)), 'dones': rng.integers(2, size=8).astype(bool)}
    copy = lambda net: [p.detach().clone() for p in net.parameters()]
    q1, q2, actor, target = copy(agent.q1), copy(agent.q2), copy(agent.actor), copy(agent.q1_t)
    log_alpha = agent.log_alpha.detach().clone()
    agent.update(batch)
    changed = lambda old, net: any(not torch.equal(o, n) for o, n in zip(old, net.parameters()))
    assert changed(q1, agent.q1) and changed(q2, agent.q2) and changed(actor, agent.actor)
    assert not torch.equal(log_alpha, agent.log_alpha.detach())
    for old, new_target, new_q in zip(target, agent.q1_t.parameters(), agent.q1.parameters()):
        torch.testing.assert_close(new_target.detach(), (1 - .01) * old + .01 * new_q.detach())


def test_sac_is_deterministic_on_cpu(scan2d, monkeypatch):
    from scRL import SACCore
    monkeypatch.setattr(SACCore, 'device', torch.device('cpu'))
    gres, Z = _grid(2, scan2d)
    runs = []
    for _ in range(2):
        seed_all(3)
        runs.append(_sac(gres, Z, 100).train()[0])
    assert runs[0] == runs[1]


def test_bad_algorithm_and_entropy_fraction(scan2d):
    gres, Z = _grid(2, scan2d)
    with pytest.raises(ValueError, match='SAC'):
        scRL.trainer('Nope', gres, reward_type='d', starts_prob=False, X_latent=Z)
    with pytest.raises(ValueError):
        scRL.trainer('SAC', gres, reward_type='d', starts_prob=False, X_latent=Z, entropy_frac=0)


@pytest.mark.parametrize('d', [2, 3])
def test_plot_policy_with_sac(d, scan2d):
    from test_plot import _arrows, _offsets, _spacing
    gres, Z = _grid(d, scan2d)
    t = _sac(gres, Z, 10)
    move, A = 1, len(_offsets(gres))
    t.agent.actor = lambda x: torch.nn.functional.one_hot(torch.full((len(x),), move, device=x.device), A).float()
    expected = _offsets(gres)[move] * _spacing(gres) * (.8 if d == 3 else 1)
    arrows = _arrows(scRL.plot_policy(gres, t, every=1), d == 3)
    assert len(arrows) == len(gres.grids['mapped_travel'])
    np.testing.assert_allclose(arrows, np.tile(expected, (len(arrows), 1)), atol=1e-6)


def test_sac_update_terms_by_hand(monkeypatch):
    import copy
    agent = _agent(monkeypatch)
    rng = np.random.default_rng(5)
    batch = {'states': rng.normal(size=(6, 5)), 'actions': np.array([0, 1, 2, 3, 4, 5]),
             'rewards': np.array([1., -2., .5, 0., 3., -1.]), 'next_states': rng.normal(size=(6, 5)),
             'dones': np.array([True, False, False, True, False, False])}
    with torch.no_grad():                                                  # targets that differ from the online nets
        for net in (agent.q1_t, agent.q2_t):
            for param in net.parameters():
                param.add_(.1)
    before = copy.deepcopy(agent)
    used = agent.update(batch)
    for target, online, old in ((agent.q1_t, agent.q1, before.q1_t), (agent.q2_t, agent.q2, before.q2_t)):
        for t_new, o_new, t_old in zip(target.parameters(), online.parameters(), old.parameters()):
            torch.testing.assert_close(t_new.detach(), (1 - agent.tau) * t_old.detach() + agent.tau * o_new.detach())
    s, s2 = torch.tensor(batch['states'], dtype=torch.float), torch.tensor(batch['next_states'], dtype=torch.float)
    r = torch.tensor(batch['rewards'], dtype=torch.float).view(-1, 1)
    d = torch.tensor(batch['dones'], dtype=torch.float).view(-1, 1)
    alpha = before.log_alpha.exp().detach()
    with torch.no_grad():
        p2 = before.actor(s2)                                              # policy expectation, not a max
        q_target = torch.minimum(before.q1_t(s2), before.q2_t(s2))         # target networks, not the online ones
        v2 = (p2 * (q_target - alpha * torch.log(p2 + 1e-8))).sum(1, keepdim=True)
        y = r + .9 * (1 - d) * v2                                          # no future after a terminal step
        q_online = torch.minimum(before.q1(s2), before.q2(s2))
        y_online = r + .9 * (1 - d) * (p2 * (q_online - alpha * torch.log(p2 + 1e-8))).sum(1, keepdim=True)
    torch.testing.assert_close(used['target'], y)
    assert not torch.allclose(y, y_online)
    assert not torch.allclose(y, r + .9 * v2)
    with torch.no_grad():
        p = before.actor(s)
        logp = torch.log(p + 1e-8)
        q = torch.minimum(agent.q1(s), agent.q2(s))                        # Q networks after their own step
        actor_loss = (p * (alpha * logp - q)).sum(1).mean()
        entropy = -(p * logp).sum(1).mean()
        alpha_loss = before.log_alpha * (entropy - before.target_entropy)  # sign: alpha grows when entropy is low
    torch.testing.assert_close(used['actor_loss'], actor_loss)
    torch.testing.assert_close(used['alpha_loss'], alpha_loss.detach())
    la = before.log_alpha.detach().clone().requires_grad_(True)
    (la * (torch.tensor(0.0) - before.target_entropy)).backward()
    assert la.grad.item() < 0                                              # too little entropy: alpha is raised
