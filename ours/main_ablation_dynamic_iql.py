import os
import random
import time
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from collections import deque
import numpy as np
import torch
from tqdm import trange
from datasets import load_d4rl_dataset
from utils.replay_buffer import ReplayBuffer
from agents.iql import IQLAgent
from evaluate import evaluate_policy
from ours.dynamic_rnd import DynamicRND
from ours.beta_scheduler import BetaScheduler

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def to_column(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return x

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def read_last_step_from_csv(csv_path: Path) -> Optional[int]:
    if not csv_path.exists():
        return None
    if csv_path.stat().st_size <= 0:
        return None
    try:
        last_row = None
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                last_row = row
        if last_row is None:
            return None
        return int(float(last_row['step']))
    except Exception:
        return None

@dataclass
class ExperimentConfig:
    name: str
    use_rnd: bool
    beta_mode: str = 'dynamic'
    fixed_beta: float = 0.0001
    use_rnd_gate: bool = True
    use_td_gate: bool = True
    use_warmup: bool = True
    warmup_steps: int = 50000
    rnd_freeze_after: int = -1
    rnd_frozen: bool = False
    potential_quantile: float = 0.6
    potential_high_quantile: float = 1.0
    rnd_low_quantile: float = 0.25
    rnd_high_quantile: float = 0.9
DATASETS = ['hopper-medium-replay-v2']
ACTOR_FREEZE_BY_ENV = {'halfcheetah-medium-v2': False, 'halfcheetah-medium-replay-v2': False, 'halfcheetah-medium-expert-v2': False, 'hopper-medium-replay-v2': False, 'hopper-medium-v2': False, 'hopper-medium-expert-v2': False, 'walker2d-medium-v2': False, 'walker2d-medium-replay-v2': False, 'walker2d-medium-expert-v2': False}
ACTOR_FREEZE_AFTER_STEP_BY_ENV = {'hopper-medium-replay-v2': 100000, 'hopper-medium-v2': 100000, 'hopper-medium-expert-v2': 100000}
ACTOR_LOSS_STABILITY_WINDOW = 20
ACTOR_LOSS_REL_STD_THRESHOLD = 0.03
CRITIC_STABILITY_THRESHOLD = 0.9
ACTOR_FREEZE_PATIENCE = 3
EXPERIMENTS = [ExperimentConfig(name='full_dynamic_beta', use_rnd=True, beta_mode='dynamic', fixed_beta=0.0001, use_rnd_gate=True, use_td_gate=True, use_warmup=True, warmup_steps=50000, rnd_freeze_after=-1, potential_quantile=0.6, rnd_low_quantile=0.25, rnd_high_quantile=0.9)]
SEEDS = [2]
MAX_TRAIN_STEPS = 500000
STEP_TAG = f'step{MAX_TRAIN_STEPS:_}'
BATCH_SIZE = 256
LOG_INTERVAL = 1000
EVAL_INTERVAL = 10000
EVAL_EPISODES = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ABLATION_NAME = 'ablation_dynamic_rnd_iql'
LOG_ROOT = os.path.join('ours_logs', ABLATION_NAME)
CKPT_ROOT = os.path.join('ours_checkpoints', ABLATION_NAME)
DONE_ROOT = os.path.join('ours_done', ABLATION_NAME)
ensure_dir(LOG_ROOT)
ensure_dir(CKPT_ROOT)
ensure_dir(DONE_ROOT)
NORMALIZE_ENV_REWARD = False
RND_HIDDEN_DIM = 256
RND_OUTPUT_DIM = 128
RND_LR = 5e-06
RND_REWARD_NORMALIZE = True
RND_REWARD_CLIP = 5.0
CURIOSITY_SCALE = 1.0
BETA_MIN = 0.0
BETA_MAX = 0.06
Q_LOSS_WINDOW = 50
Q_LOSS_STABLE_STD = 10
Q_LOSS_UNSTABLE_STD = 40
ABNORMAL_Q_LOSS = 200.0
MIN_GATE = 0.0
VALUE_GATE_START_STEP = 50000
TD_GATE_DISCOUNT = 0.99

def get_exp_id(exp_name: str, env_name: str, seed: int) -> str:
    return f'{exp_name}_{env_name}_seed{seed}_{STEP_TAG}'

def get_log_paths(exp_name: str, env_name: str, seed: int) -> Tuple[str, str]:
    exp_id = get_exp_id(exp_name, env_name, seed)
    train_log_path = os.path.join(LOG_ROOT, f'train_log_{exp_id}.csv')
    eval_log_path = os.path.join(LOG_ROOT, f'eval_log_{exp_id}.csv')
    return (train_log_path, eval_log_path)

def get_checkpoint_paths(exp_name: str, env_name: str, seed: int) -> Tuple[str, str]:
    exp_id = get_exp_id(exp_name, env_name, seed)
    exp_ckpt_dir = os.path.join(CKPT_ROOT, exp_name)
    ensure_dir(exp_ckpt_dir)
    latest_save_path = os.path.join(exp_ckpt_dir, f'{exp_id}_latest.pt')
    best_save_path = os.path.join(exp_ckpt_dir, f'{exp_id}_best.pt')
    return (latest_save_path, best_save_path)

def get_done_path(exp_name: str, env_name: str, seed: int) -> Path:
    exp_id = get_exp_id(exp_name, env_name, seed)
    return Path(DONE_ROOT) / f'{exp_id}.done'

def get_legacy_eval_candidates(exp_name: str, env_name: str, seed: int) -> List[Path]:
    candidates = []
    if exp_name == 'full':
        candidates.extend([Path('ours_logs') / 'ours_dynamic_rnd_iql' / f'eval_log_ours_dynamic_rnd_iql_{env_name}_seed{seed}_{STEP_TAG}.csv', Path(f'eval_log_ours_dynamic_rnd_iql_{env_name}_seed{seed}_{STEP_TAG}.csv')])
    if exp_name == 'iql_baseline':
        candidates.extend([Path(f'eval_log_{env_name}_seed{seed}_{STEP_TAG}.csv'), Path('logs') / f'eval_log_{env_name}_seed{seed}_{STEP_TAG}.csv', Path('iql_logs') / f'eval_log_{env_name}_seed{seed}_{STEP_TAG}.csv', Path('logs') / 'iql' / f'eval_log_iql_{env_name}_seed{seed}_{STEP_TAG}.csv', Path('iql_logs') / 'iql' / f'eval_log_iql_{env_name}_seed{seed}_{STEP_TAG}.csv'])
    return candidates

def experiment_finished(exp_name: str, env_name: str, seed: int) -> Tuple[bool, str]:
    done_path = get_done_path(exp_name, env_name, seed)
    if done_path.exists():
        return (True, f'found done file: {done_path}')
    _, eval_log_path = get_log_paths(exp_name, env_name, seed)
    eval_log_path = Path(eval_log_path)
    last_step = read_last_step_from_csv(eval_log_path)
    if last_step is not None and last_step >= MAX_TRAIN_STEPS:
        return (True, f'found completed eval log: {eval_log_path}')
    for legacy_path in get_legacy_eval_candidates(exp_name, env_name, seed):
        legacy_last_step = read_last_step_from_csv(legacy_path)
        if legacy_last_step is not None and legacy_last_step >= MAX_TRAIN_STEPS:
            return (True, f'found completed legacy eval log: {legacy_path}')
    return (False, 'no completed result found')

def write_done_file(exp_name: str, env_name: str, seed: int, best_step: int, best_score: float):
    done_path = get_done_path(exp_name, env_name, seed)
    with open(done_path, mode='w', encoding='utf-8') as f:
        f.write(f'experiment: {exp_name}\n')
        f.write(f'env_name: {env_name}\n')
        f.write(f'seed: {seed}\n')
        f.write(f'max_train_steps: {MAX_TRAIN_STEPS}\n')
        f.write(f'best_step: {best_step}\n')
        f.write(f'best_score: {best_score}\n')

def create_csv_logs(train_log_path: str, eval_log_path: str):
    with open(train_log_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'value_loss', 'critic_loss', 'actor_loss', 'mean_advantage', 'env_reward_mean', 'curiosity_reward_mean', 'mixed_reward_mean', 'mask_ratio', 'rnd_loss', 'rnd_error_mean', 'rnd_error_std', 'curiosity_reward_std', 'beta', 'stability', 'rnd_score', 'q_loss_mean', 'q_loss_std', 'elapsed_time'])
    with open(eval_log_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'average_return', 'normalized_score', 'best_score', 'beta', 'stability', 'rnd_score'])

def write_train_log(train_log_path: str, step: int, value_loss: float, critic_loss: float, actor_loss: float, mean_advantage: float, env_reward_mean: float, curiosity_reward_mean: float, mixed_reward_mean: float, mask_ratio: float, rnd_loss: float, rnd_error_mean: float, rnd_error_std: float, curiosity_reward_std: float, beta: float, stability: float, rnd_score: float, q_loss_mean, q_loss_std, elapsed_time: float):
    with open(train_log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([step, value_loss, critic_loss, actor_loss, mean_advantage, env_reward_mean, curiosity_reward_mean, mixed_reward_mean, mask_ratio, rnd_loss, rnd_error_mean, rnd_error_std, curiosity_reward_std, beta, stability, rnd_score, q_loss_mean, q_loss_std, elapsed_time])

def write_eval_log(eval_log_path: str, step: int, average_return: float, normalized_score: float, best_score: float, beta: float, stability: float, rnd_score: float):
    with open(eval_log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([step, average_return, normalized_score, best_score, beta, stability, rnd_score])

def save_checkpoint(agent: IQLAgent, dynamic_rnd: Optional[DynamicRND], beta_scheduler: BetaScheduler, save_path: str, step: int, exp_config: ExperimentConfig):
    checkpoint = {'step': step, 'experiment': exp_config.name, 'actor': agent.actor.state_dict(), 'critic': agent.critic.state_dict(), 'value': agent.value.state_dict(), 'target_critic': agent.target_critic.state_dict(), 'beta_scheduler': beta_scheduler.state_dict(), 'exp_config': {'name': exp_config.name, 'use_rnd': exp_config.use_rnd, 'beta_mode': exp_config.beta_mode, 'fixed_beta': exp_config.fixed_beta, 'use_rnd_gate': exp_config.use_rnd_gate, 'use_td_gate': exp_config.use_td_gate, 'use_warmup': exp_config.use_warmup, 'warmup_steps': exp_config.warmup_steps, 'rnd_freeze_after': exp_config.rnd_freeze_after, 'potential_quantile': exp_config.potential_quantile, 'potential_high_quantile': exp_config.potential_high_quantile, 'rnd_low_quantile': exp_config.rnd_low_quantile, 'rnd_high_quantile': exp_config.rnd_high_quantile}}
    if dynamic_rnd is not None:
        checkpoint.update({'rnd_target': dynamic_rnd.target.state_dict(), 'rnd_predictor': dynamic_rnd.predictor.state_dict(), 'rnd_optimizer': dynamic_rnd.optimizer.state_dict(), 'rnd_reward_rms_mean': dynamic_rnd.reward_rms.mean, 'rnd_reward_rms_var': dynamic_rnd.reward_rms.var, 'rnd_reward_rms_count': dynamic_rnd.reward_rms.count})
    torch.save(checkpoint, save_path)

def freeze_rnd_optimizer(dynamic_rnd: DynamicRND):
    for param_group in dynamic_rnd.optimizer.param_groups:
        param_group['lr'] = 0.0

def maybe_freeze_rnd(exp_config: ExperimentConfig, dynamic_rnd: Optional[DynamicRND], step: int):
    if dynamic_rnd is None:
        return
    if not exp_config.use_rnd:
        return
    if exp_config.rnd_freeze_after < 0:
        return
    if exp_config.rnd_frozen:
        return
    if step >= exp_config.rnd_freeze_after:
        freeze_rnd_optimizer(dynamic_rnd)
        exp_config.rnd_frozen = True
        print(f'\n[RND Frozen] Experiment: {exp_config.name} | Step: {step} | RND optimizer lr has been set to 0.')

def get_beta_for_reward(exp_config: ExperimentConfig, beta_scheduler: BetaScheduler, step: int) -> float:
    if not exp_config.use_rnd:
        return 0.0
    if exp_config.use_warmup and step < exp_config.warmup_steps:
        return 0.0
    if exp_config.beta_mode == 'fixed':
        beta = exp_config.fixed_beta
    elif exp_config.beta_mode == 'dynamic':
        beta = beta_scheduler.get_beta()
    else:
        beta = 0.0
    return float(beta)

def build_mixed_rewards(exp_config: ExperimentConfig, rewards: torch.Tensor, curiosity_reward: torch.Tensor, beta_for_reward: float, agent: IQLAgent, observations: torch.Tensor, actions: torch.Tensor, next_observations: torch.Tensor, terminals: torch.Tensor, step: int) -> Tuple[torch.Tensor, float]:
    if not exp_config.use_rnd or beta_for_reward <= 0.0:
        mixed_rewards = rewards
        mask_ratio = 0.0
        return (mixed_rewards, mask_ratio)
    with torch.no_grad():
        rnd = curiosity_reward.detach()
        rnd = torch.clamp(rnd, min=0.0)
        rnd = rnd / (rnd.mean() + 1e-06)
        if exp_config.use_rnd_gate:
            rnd_flat = rnd.view(-1)
            rnd_low = torch.quantile(rnd_flat, exp_config.rnd_low_quantile)
            rnd_high = torch.quantile(rnd_flat, exp_config.rnd_high_quantile)
            rnd_mask = ((rnd > rnd_low) & (rnd <= rnd_high)).float()
        else:
            rnd_mask = torch.ones_like(rewards)
        if exp_config.use_td_gate and step >= VALUE_GATE_START_STEP:
            v = agent.value(observations)
            next_v = agent.value(next_observations)
            v = to_column(v)
            next_v = to_column(next_v)
            not_done = 1.0 - terminals
            potential = rewards + TD_GATE_DISCOUNT * not_done * next_v - v
            potential_flat = potential.detach().view(-1)
            potential_low = torch.quantile(potential_flat, exp_config.potential_quantile)
            potential_high = torch.quantile(potential_flat, exp_config.potential_high_quantile)
            potential_mask = ((potential.detach() > potential_low) & (potential.detach() <= potential_high) & (potential.detach() > 0.0)).float()
        else:
            potential_mask = torch.ones_like(rewards)
        reward_mask = rnd_mask * potential_mask
        mask_ratio = float(reward_mask.mean().item())
        raw_bonus = beta_for_reward * CURIOSITY_SCALE * rnd * reward_mask
        bonus = raw_bonus
    mixed_rewards = rewards + bonus
    return (mixed_rewards, mask_ratio)

def run_one_experiment(env_name: str, exp_config: ExperimentConfig, seed: int):
    exp_name = exp_config.name
    already_done, reason = experiment_finished(exp_name=exp_name, env_name=env_name, seed=seed)
    if already_done:
        print('=' * 80)
        print(f'[Skip] Experiment already finished')
        print(f'Env: {env_name}')
        print(f'Exp: {exp_name}')
        print(f'Reason: {reason}')
        print('=' * 80)
        return
    exp_config.rnd_frozen = False
    train_log_path, eval_log_path = get_log_paths(exp_name=exp_name, env_name=env_name, seed=seed)
    latest_save_path, best_save_path = get_checkpoint_paths(exp_name=exp_name, env_name=env_name, seed=seed)
    create_csv_logs(train_log_path=train_log_path, eval_log_path=eval_log_path)
    set_seed(seed)
    print('=' * 80)
    print('Ablation Dynamic RND-IQL Training')
    print('=' * 80)
    print('Environment:', env_name)
    print('Experiment:', exp_name)
    print('Seed:', seed)
    print('Device:', DEVICE)
    print('Max train steps:', MAX_TRAIN_STEPS)
    print('Batch size:', BATCH_SIZE)
    print('Use RND:', exp_config.use_rnd)
    print('Beta mode:', exp_config.beta_mode)
    print('Fixed beta:', exp_config.fixed_beta)
    print('Use RND gate:', exp_config.use_rnd_gate)
    print('Use TD gate:', exp_config.use_td_gate)
    print('Use warmup:', exp_config.use_warmup)
    print('Warmup steps:', exp_config.warmup_steps)
    print('RND freeze after:', exp_config.rnd_freeze_after)
    print('BETA_MIN:', BETA_MIN)
    print('BETA_MAX:', BETA_MAX)
    print('CURIOSITY_SCALE:', CURIOSITY_SCALE)
    print('MIN_GATE:', MIN_GATE)
    print('Potential quantile:', exp_config.potential_quantile)
    print('RND low quantile:', exp_config.rnd_low_quantile)
    print('RND high quantile:', exp_config.rnd_high_quantile)
    print('Train log path:', train_log_path)
    print('Eval log path:', eval_log_path)
    print('Latest checkpoint path:', latest_save_path)
    print('Best checkpoint path:', best_save_path)
    print('=' * 80)
    data, info = load_d4rl_dataset(env_name, normalize_reward=NORMALIZE_ENV_REWARD)
    env = info['env']
    observation_dim = info['observation_dim']
    action_dim = info['action_dim']
    max_action = info['max_action']
    print('Observation dim:', observation_dim)
    print('Action dim:', action_dim)
    print('Max action:', max_action)
    print('Dataset size:', info['dataset_size'])
    print('=' * 80)
    try:
        env.seed(seed)
        env.action_space.seed(seed)
    except Exception:
        pass
    replay_buffer = ReplayBuffer(data=data, device=DEVICE)
    agent = IQLAgent(observation_dim=observation_dim, action_dim=action_dim, max_action=max_action, device=DEVICE, hidden_dim=256, hidden_depth=2, discount=0.99, expectile=0.7, temperature=3.0, tau=0.005, actor_lr=0.0003, critic_lr=0.0003, value_lr=0.0003, max_steps=MAX_TRAIN_STEPS, use_actor_lr_decay=True, grad_clip_norm=10.0)
    if exp_config.use_rnd:
        dynamic_rnd = DynamicRND(state_dim=observation_dim, action_dim=action_dim, device=torch.device(DEVICE), hidden_dim=RND_HIDDEN_DIM, output_dim=RND_OUTPUT_DIM, lr=RND_LR, reward_normalize=RND_REWARD_NORMALIZE, reward_clip=RND_REWARD_CLIP, input_normalize=False)
    else:
        dynamic_rnd = None
    beta_scheduler = BetaScheduler(beta_min=BETA_MIN, beta_max=BETA_MAX, q_loss_window=Q_LOSS_WINDOW, q_loss_stable_std=Q_LOSS_STABLE_STD, q_loss_unstable_std=Q_LOSS_UNSTABLE_STD, abnormal_q_loss=ABNORMAL_Q_LOSS)
    print('Beta scheduler initialized:')
    beta_scheduler.print_info()
    print('Evaluating untrained policy...')
    eval_result = evaluate_policy(env=env, agent=agent, device=torch.device(DEVICE), eval_episodes=EVAL_EPISODES, deterministic=True)
    current_score = eval_result['normalized_score']
    scheduler_state = beta_scheduler.state_dict()
    beta_for_log = 0.0
    if exp_config.use_rnd:
        if exp_config.beta_mode == 'fixed':
            beta_for_log = exp_config.fixed_beta
        elif exp_config.beta_mode == 'dynamic':
            beta_for_log = scheduler_state['beta']
    print(f"[Before Training] Average Return: {eval_result['average_return']:.2f} | Normalized Score: {current_score:.2f}")
    best_normalized_score = current_score
    best_step = 0
    write_eval_log(eval_log_path=eval_log_path, step=0, average_return=eval_result['average_return'], normalized_score=current_score, best_score=best_normalized_score, beta=beta_for_log, stability=scheduler_state['stability'], rnd_score=scheduler_state['rnd_score'])
    save_checkpoint(agent=agent, dynamic_rnd=dynamic_rnd, beta_scheduler=beta_scheduler, save_path=best_save_path, step=0, exp_config=exp_config)
    print(f'Initial best checkpoint saved to: {best_save_path} | Best Score: {best_normalized_score:.2f}')
    print('=' * 80)
    start_time = time.time()
    use_actor_freeze = ACTOR_FREEZE_BY_ENV.get(env_name, False)
    actor_freeze_after_step = ACTOR_FREEZE_AFTER_STEP_BY_ENV.get(env_name, 200000)
    actor_already_frozen = False
    actor_loss_stability_window = deque(maxlen=ACTOR_LOSS_STABILITY_WINDOW)
    actor_freeze_stable_count = 0
    recent_value_losses = []
    recent_critic_losses = []
    recent_actor_losses = []
    recent_advantages = []
    recent_env_reward_means = []
    recent_curiosity_reward_means = []
    recent_mixed_reward_means = []
    recent_mask_ratios = []
    recent_rnd_losses = []
    recent_rnd_error_means = []
    recent_rnd_error_stds = []
    recent_curiosity_reward_stds = []
    recent_betas = []
    recent_rnd_scores = []
    recent_stabilities = []
    for step in trange(1, MAX_TRAIN_STEPS + 1, desc=f'Training {exp_name} on {env_name}'):
        batch = replay_buffer.sample(batch_size=BATCH_SIZE)
        observations = batch['observations']
        actions = batch['actions']
        rewards = to_column(batch['rewards'])
        next_observations = batch['next_observations']
        terminals = to_column(batch['terminals'])
        maybe_freeze_rnd(exp_config=exp_config, dynamic_rnd=dynamic_rnd, step=step)
        if exp_config.use_rnd:
            curiosity_reward, rnd_info = dynamic_rnd.update_and_get_reward(states=observations, actions=actions, next_states=next_observations)
            curiosity_reward = to_column(curiosity_reward)
        else:
            curiosity_reward = torch.zeros_like(rewards)
            rnd_info = {'rnd_loss': 0.0, 'rnd_error_mean': 0.0, 'rnd_error_std': 0.0, 'curiosity_reward_std': 0.0}
        beta_for_reward = get_beta_for_reward(exp_config=exp_config, beta_scheduler=beta_scheduler, step=step)
        mixed_rewards, mask_ratio = build_mixed_rewards(exp_config=exp_config, rewards=rewards, curiosity_reward=curiosity_reward, beta_for_reward=beta_for_reward, agent=agent, observations=observations, actions=actions, next_observations=next_observations, terminals=terminals, step=step)
        train_batch = {'observations': observations, 'actions': actions, 'rewards': mixed_rewards, 'next_observations': next_observations, 'terminals': terminals}
        logs = agent.update(train_batch)
        critic_loss = logs['critic_loss']
        beta_scheduler.update(q_loss=critic_loss, rnd_error=rnd_info['rnd_error_mean'])
        scheduler_state = beta_scheduler.state_dict()
        if exp_config.use_rnd:
            if exp_config.beta_mode == 'fixed':
                beta_for_log = exp_config.fixed_beta
            else:
                beta_for_log = beta_for_reward
        else:
            beta_for_log = 0.0
        recent_value_losses.append(logs['value_loss'])
        recent_critic_losses.append(logs['critic_loss'])
        recent_actor_losses.append(logs['actor_loss'])
        recent_advantages.append(logs['mean_advantage'])
        recent_env_reward_means.append(float(rewards.detach().mean().item()))
        recent_curiosity_reward_means.append(float(curiosity_reward.detach().mean().item()))
        recent_mixed_reward_means.append(float(mixed_rewards.detach().mean().item()))
        recent_mask_ratios.append(mask_ratio)
        recent_rnd_losses.append(float(rnd_info['rnd_loss']))
        recent_rnd_error_means.append(float(rnd_info['rnd_error_mean']))
        recent_rnd_error_stds.append(float(rnd_info['rnd_error_std']))
        recent_curiosity_reward_stds.append(float(rnd_info['curiosity_reward_std']))
        recent_betas.append(float(beta_for_log))
        recent_rnd_scores.append(float(scheduler_state['rnd_score']))
        recent_stabilities.append(float(scheduler_state['stability']))
        if step % LOG_INTERVAL == 0:
            mean_value_loss = float(np.mean(recent_value_losses))
            mean_critic_loss = float(np.mean(recent_critic_losses))
            mean_actor_loss = float(np.mean(recent_actor_losses))
            mean_advantage = float(np.mean(recent_advantages))
            mean_env_reward = float(np.mean(recent_env_reward_means))
            mean_curiosity_reward = float(np.mean(recent_curiosity_reward_means))
            mean_mixed_reward = float(np.mean(recent_mixed_reward_means))
            mean_mask_ratio = float(np.mean(recent_mask_ratios))
            mean_rnd_loss = float(np.mean(recent_rnd_losses))
            mean_rnd_error = float(np.mean(recent_rnd_error_means))
            mean_rnd_error_std = float(np.mean(recent_rnd_error_stds))
            mean_curiosity_reward_std = float(np.mean(recent_curiosity_reward_stds))
            mean_beta = float(np.mean(recent_betas))
            mean_rnd_score = float(np.mean(recent_rnd_scores))
            mean_stability = float(np.mean(recent_stabilities))
            actor_loss_stability_window.append(mean_actor_loss)
            actor_loss_relative_std = None
            if len(actor_loss_stability_window) == ACTOR_LOSS_STABILITY_WINDOW:
                actor_loss_window_mean = float(np.mean(actor_loss_stability_window))
                actor_loss_window_std = float(np.std(actor_loss_stability_window))
                actor_loss_relative_std = actor_loss_window_std / (abs(actor_loss_window_mean) + 1e-08)
                actor_is_stable = actor_loss_relative_std <= ACTOR_LOSS_REL_STD_THRESHOLD
                critic_is_stable = scheduler_state['q_loss_window_ready'] and mean_stability >= CRITIC_STABILITY_THRESHOLD
                reached_minimum_step = step >= actor_freeze_after_step
                if use_actor_freeze and (not actor_already_frozen) and reached_minimum_step and actor_is_stable and critic_is_stable:
                    actor_freeze_stable_count += 1
                else:
                    actor_freeze_stable_count = 0
                if use_actor_freeze and (not actor_already_frozen) and (actor_freeze_stable_count >= ACTOR_FREEZE_PATIENCE):
                    agent.freeze_actor()
                    actor_already_frozen = True
                    print(f'\nActor frozen | Env: {env_name} | Exp: {exp_name} | Step: {step} | Actor Loss Relative Std: {actor_loss_relative_std:.6f} | Critic Stability: {mean_stability:.4f}')
            q_loss_mean = scheduler_state['q_loss_mean']
            q_loss_std = scheduler_state['q_loss_std']
            elapsed_time = float(time.time() - start_time)
            print(f'\nStep: {step} | Env: {env_name} | Exp: {exp_name} | V Loss: {mean_value_loss:.4f} | Critic Loss: {mean_critic_loss:.4f} | Actor Loss: {mean_actor_loss:.4f} | Adv: {mean_advantage:.4f} | Beta: {mean_beta:.8f} | CurReward: {mean_curiosity_reward:.4f} | Mask: {mean_mask_ratio:.4f} | RND Score: {mean_rnd_score:.4f} | Actor Frozen: {actor_already_frozen} | RND Loss: {mean_rnd_loss:.4f} | Time: {elapsed_time:.1f}s')
            write_train_log(train_log_path=train_log_path, step=step, value_loss=mean_value_loss, critic_loss=mean_critic_loss, actor_loss=mean_actor_loss, mean_advantage=mean_advantage, env_reward_mean=mean_env_reward, curiosity_reward_mean=mean_curiosity_reward, mixed_reward_mean=mean_mixed_reward, mask_ratio=mean_mask_ratio, rnd_loss=mean_rnd_loss, rnd_error_mean=mean_rnd_error, rnd_error_std=mean_rnd_error_std, curiosity_reward_std=mean_curiosity_reward_std, beta=mean_beta, stability=mean_stability, rnd_score=mean_rnd_score, q_loss_mean=q_loss_mean, q_loss_std=q_loss_std, elapsed_time=elapsed_time)
            recent_value_losses.clear()
            recent_critic_losses.clear()
            recent_actor_losses.clear()
            recent_advantages.clear()
            recent_env_reward_means.clear()
            recent_curiosity_reward_means.clear()
            recent_mixed_reward_means.clear()
            recent_mask_ratios.clear()
            recent_rnd_losses.clear()
            recent_rnd_error_means.clear()
            recent_rnd_error_stds.clear()
            recent_curiosity_reward_stds.clear()
            recent_betas.clear()
            recent_rnd_scores.clear()
            recent_stabilities.clear()
        if step % EVAL_INTERVAL == 0:
            eval_result = evaluate_policy(env=env, agent=agent, device=torch.device(DEVICE), eval_episodes=EVAL_EPISODES, deterministic=True)
            current_score = eval_result['normalized_score']
            scheduler_state = beta_scheduler.state_dict()
            if exp_config.use_rnd:
                if exp_config.beta_mode == 'fixed':
                    beta_for_eval_log = exp_config.fixed_beta
                else:
                    beta_for_eval_log = beta_scheduler.get_beta()
            else:
                beta_for_eval_log = 0.0
            print(f"\n[Evaluation] Env: {env_name} | Exp: {exp_name} | Step: {step} | Average Return: {eval_result['average_return']:.2f} | Normalized Score: {current_score:.2f} | Best Score: {best_normalized_score:.2f} | Beta: {beta_for_eval_log:.8f} | RND Score: {scheduler_state['rnd_score']:.4f} | Stability: {scheduler_state['stability']:.4f}")
            write_eval_log(eval_log_path=eval_log_path, step=step, average_return=eval_result['average_return'], normalized_score=current_score, best_score=best_normalized_score, beta=beta_for_eval_log, stability=scheduler_state['stability'], rnd_score=scheduler_state['rnd_score'])
            save_checkpoint(agent=agent, dynamic_rnd=dynamic_rnd, beta_scheduler=beta_scheduler, save_path=latest_save_path, step=step, exp_config=exp_config)
            print(f'Latest checkpoint saved to: {latest_save_path}')
            if current_score > best_normalized_score:
                best_normalized_score = current_score
                best_step = step
                save_checkpoint(agent=agent, dynamic_rnd=dynamic_rnd, beta_scheduler=beta_scheduler, save_path=best_save_path, step=step, exp_config=exp_config)
                print(f'New best checkpoint saved to: {best_save_path} | Best Step: {best_step} | Best Score: {best_normalized_score:.2f}')
            else:
                print(f'No improvement | Best Step: {best_step} | Best Score: {best_normalized_score:.2f}')
    save_checkpoint(agent=agent, dynamic_rnd=dynamic_rnd, beta_scheduler=beta_scheduler, save_path=latest_save_path, step=MAX_TRAIN_STEPS, exp_config=exp_config)
    write_done_file(exp_name=exp_name, env_name=env_name, seed=seed, best_step=best_step, best_score=best_normalized_score)
    print('=' * 80)
    print('Ablation experiment finished.')
    print('Environment:', env_name)
    print('Experiment:', exp_name)
    print(f'Final latest checkpoint saved to: {latest_save_path}')
    print(f'Best checkpoint saved to: {best_save_path}')
    print(f'Best Step: {best_step}')
    print(f'Best Normalized Score: {best_normalized_score:.2f}')
    print(f'Train log saved to: {train_log_path}')
    print(f'Eval log saved to: {eval_log_path}')
    print('=' * 80)
    try:
        env.close()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    print('=' * 80)
    print('Start all ablation experiments')
    print('=' * 80)
    print('Datasets:')
    for env_name in DATASETS:
        print('  -', env_name)
    print('Seeds:')
    for seed in SEEDS:
        print('  -', seed)
    print('Experiments:')
    for exp in EXPERIMENTS:
        print('  -', exp.name)
    print('=' * 80)
    for env_name in DATASETS:
        for seed in SEEDS:
            for exp_config in EXPERIMENTS:
                run_one_experiment(env_name=env_name, exp_config=exp_config, seed=seed)
    print('=' * 80)
    print('All ablation experiments finished.')
    print('=' * 80)
if __name__ == '__main__':
    main()
