import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from mani_skill.utils import gym_utils
from mani_skill.utils.io_utils import load_json
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
from tqdm import tqdm

from behavior_cloning.evaluate import evaluate
from behavior_cloning.make_env import make_eval_envs

import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode, CPUGymWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv


@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    env_id: str = "PegInsertionSide-v0"
    """the id of the environment"""
    demo_path: str = "data/ms2_official_demos/rigid_body/PegInsertionSide-v0/trajectory.state.pd_ee_delta_pose.h5"
    """the path of demo dataset (pkl or h5)"""
    num_demos: Optional[int] = None
    """number of trajectories to load from the demo dataset"""
    total_iters: int = 1_000_000
    """total timesteps of the experiment"""
    batch_size: int = 1024
    """the batch size of sample from the replay memory"""

    # Behavior cloning specific arguments
    lr: float = 3e-4
    """the learning rate for the actor"""
    normalize_states: bool = False
    """if toggled, states are normalized to mean 0 and standard deviation 1"""

    # Environment/experiment specific arguments
    max_episode_steps: Optional[int] = None
    """Change the environments' max_episode_steps to this value. Sometimes necessary if the demonstrations being imitated are too short. Typically the default
    max episode steps of environments in ManiSkill are tuned lower so reinforcement learning agents can learn faster."""
    ppo_freq : int = 100
    ppo_startpoint: int = 5000
    log_freq: int = 1000
    """the frequency of logging the training metrics"""
    eval_freq: int = 1000
    """the frequency of evaluating the agent on the evaluation environments"""
    save_freq: Optional[int] = None
    """the frequency of saving the model checkpoints. By default this is None and will only save checkpoints based on the best evaluation metrics."""
    num_eval_episodes: int = 100
    """the number of episodes to evaluate the agent on"""
    num_envs: int = 512
    num_eval_envs: int = 10
    partial_reset: bool = True
    num_steps: int = 50
    """the number of parallel environments to evaluate the agent on"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    finite_horizon_gae: bool = True
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    sim_backend: str = "cpu"
    """the simulation backend to use for evaluation environments. can be "cpu" or "gpu"""
    num_dataload_workers: int = 0
    """the number of workers to use for loading the training data in the torch dataloader"""
    control_mode: str = "pd_joint_delta_pos"
    """the control mode to use for the evaluation environments. Must match the control mode of the demonstration dataset."""
    ppo_num_iterations = 50
    ppo_log_freq=10
    ppo_eval_freq =10

    # additional tags/configs for logging purposes to wandb and shared comparisons with other algorithms
    demo_type: Optional[str] = None

    ppo_batch_size: int = 0
    """the batch size (computed in runtime)"""
    ppo_minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""


# taken from here
# https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Segmentation/MaskRCNN/pytorch/maskrcnn_benchmark/data/samplers/iteration_based_batch_sampler.py
class IterationBasedBatchSampler(BatchSampler):
    """
    Wraps a BatchSampler, resampling from it until
    a specified number of iterations have been sampled
    """

    def __init__(self, batch_sampler, num_iterations, start_iter=0):
        self.batch_sampler = batch_sampler
        self.num_iterations = num_iterations
        self.start_iter = start_iter

    def __iter__(self):
        iteration = self.start_iter
        while iteration <= self.num_iterations:
            # if the underlying sampler has a set_epoch method, like
            # DistributedSampler, used for making each process see
            # a different split of the dataset, then set it
            if hasattr(self.batch_sampler.sampler, "set_epoch"):
                self.batch_sampler.sampler.set_epoch(iteration)
            for batch in self.batch_sampler:
                iteration += 1
                if iteration > self.num_iterations:
                    break
                yield batch

    def __len__(self):
        return self.num_iterations


def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out


class ManiSkillDataset(Dataset):
    def __init__(
        self,
        dataset_file: str,
        device,
        load_count=-1,
        normalize_states=False,
    ) -> None:
        self.dataset_file = dataset_file
        # for details on how the code below works, see the
        # quick start tutorial
        self.data = h5py.File(dataset_file, "r")
        json_path = dataset_file.replace(".h5", ".json")
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]

        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]

        self.observations = []
        self.actions = []
        self.dones = []
        self.total_frames = 0
        self.device = device
        if load_count is None:
            load_count = len(self.episodes)
        print(f"Loading {load_count} episodes")

        for eps_id in tqdm(range(load_count)):
            eps = self.episodes[eps_id]
            trajectory = self.data[f"traj_{eps['episode_id']}"]
            trajectory = load_h5_data(trajectory)

            # we use :-1 here to ignore the last observation as that
            # is the terminal observation which has no actions
            self.observations.append(trajectory["obs"][:-1])
            self.actions.append(trajectory["actions"])
            self.dones.append(trajectory["success"].reshape(-1, 1))

        self.observations = np.vstack(self.observations)
        self.actions = np.vstack(self.actions)
        self.dones = np.vstack(self.dones)
        assert self.observations.shape[0] == self.actions.shape[0]
        assert self.dones.shape[0] == self.actions.shape[0]

        if normalize_states:
            mean, std = self.get_state_stats()
            self.observations = (self.observations - mean) / std

    def get_state_stats(self):
        return np.mean(self.observations), np.std(self.observations)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        action = torch.from_numpy(self.actions[idx]).float().to(device=self.device)
        obs = torch.from_numpy(self.observations[idx]).float().to(device=self.device)
        done = torch.from_numpy(self.dones[idx]).to(device=self.device)
        return obs, action, done

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(Actor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class Agent(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(state_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        print("obs_dimension",state_dim)
        print("action_dimension", action_dim)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(state_dim).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01*np.sqrt(2)),
        )
        # state_dict = torch.load('./prior/best_eval_success_at_end.pt')
        # self.actor.load_state_dict(state_dict)
        self.actor_logstd = nn.Parameter(torch.ones(1, action_dim) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        action_mean = self.actor(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)


def save_ckpt(run_name, tag):
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
    torch.save(
        actor.actor.state_dict(),
        f"runs/{run_name}/checkpoints/{tag}.pt",
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    args.ppo_batch_size = int(args.num_envs * args.num_steps)
    args.ppo_minibatch_size = int(args.ppo_batch_size // args.num_minibatches)

    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name

    if args.demo_path.endswith(".h5"):
        import json

        json_file = args.demo_path[:-2] + "json"
        with open(json_file, "r") as f:
            demo_info = json.load(f)
            if "control_mode" in demo_info["env_info"]["env_kwargs"]:
                control_mode = demo_info["env_info"]["env_kwargs"]["control_mode"]
            elif "control_mode" in demo_info["episodes"][0]:
                control_mode = demo_info["episodes"][0]["control_mode"]
            else:
                raise Exception("Control mode not found in json")
            assert (
                control_mode == args.control_mode
            ), f"Control mode mismatched. Dataset has control mode {control_mode}, but args has control mode {args.control_mode}"

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # bc_envs = gym.make(args.env_id, reconfiguration_freq=1, **env_kwargs)
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs , reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    # bc_envs = CPUGymWrapper(bc_envs, ignore_terminations=True, record_metrics=True)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    # bc_envs = RecordEpisode(bc_envs, output_dir=f"runs/{run_name}/videos", save_trajectory=False, info_on_video=True, source_type="behavior_cloning", source_desc="behavior_cloning evaluation rollout")
    # bc_envs.action_space.seed(seed)
    # bc_envs.observation_space.seed(seed)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/ppo_videos"
        # if args.evaluate:
        #     eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        # if args.save_train_video_freq is not None:
        #     save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
        #     envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=False, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    # vector_cls = gym.vector.SyncVectorEnv if args.num_eval_envs == 1 else lambda x : gym.vector.AsyncVectorEnv(x, context="forkserver")
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=True, record_metrics=True)

    env_kwargs = dict(
        control_mode=args.control_mode,
        reward_mode="sparse",
        obs_mode="state",
        render_mode="rgb_array",
    )
    if args.max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_episode_steps
    bc_envs = make_eval_envs(
        args.env_id,
        args.num_eval_envs,
        args.sim_backend,
        env_kwargs,
        video_dir=f"runs/{run_name}/videos" if args.capture_video else None,
    )



    if args.track:
        import wandb

        config = vars(args)
        config["eval_env_cfg"] = dict(
            **env_kwargs,
            num_envs=args.num_eval_envs,
            env_id=args.env_id,
            env_horizon=gym_utils.find_max_episode_steps_value(bc_envs),
        )
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=config,
            name=run_name,
            save_code=True,
            group="BehaviorCloning",
            tags=["behavior_cloning"],
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    ds = ManiSkillDataset(
        args.demo_path,
        device=device,
        load_count=args.num_demos,
        normalize_states=args.normalize_states,
    )

    bc_obs, _ = bc_envs.reset(seed=args.seed)

    sampler = RandomSampler(ds)
    batchsampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    itersampler = IterationBasedBatchSampler(batchsampler, args.total_iters)
    dataloader = DataLoader(
        ds, batch_sampler=itersampler, num_workers=args.num_dataload_workers
    )
    actor = Agent(
        bc_envs.single_observation_space.shape[0], bc_envs.single_action_space.shape[0]
    )
    actor = actor.to(device=device)
    # optimizer_bc = optim.Adam(actor.parameters(), lr=args.lr)
    optimizer = optim.Adam(actor.parameters(), lr=args.lr, eps=1e-5)

    best_eval_metrics = defaultdict(float)

    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    eps_returns = torch.zeros(args.num_envs, dtype=torch.float, device=device)
    print(f"####")
    print(f"args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.ppo_minibatch_size} args.batch_size={args.ppo_batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    for iteration, batch in enumerate(dataloader):
        # print("bc training start")
        log_dict = {}
        bc_obs, action, _ = batch
        pred_action = actor(bc_obs)
        optimizer.zero_grad()
        loss = F.mse_loss(pred_action, action)
        # loss.requires_grad = True
        loss.backward()
        optimizer.step()
        # print("bc training done")

        if(iteration % args.ppo_freq == args.ppo_freq-1 and iteration > args.ppo_startpoint):
            final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
            actor.eval()

            # if iteration % args.ppo_eval_freq == 1:
            #     print("Evaluating")
            #     eval_obs, _ = eval_envs.reset()
            #     eval_metrics = defaultdict(list)
            #     num_episodes = 0
            #     final_info = {}
            #     for _ in range(args.num_eval_steps):
            #         with torch.no_grad():
            #             eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_action(eval_obs, deterministic=True))
            #             if "final_info" in eval_infos:
            #                 mask = eval_infos["_final_info"]
            #                 num_episodes += mask.sum()
            #                 for k, v in eval_infos["final_info"]["episode"].items():
            #                     eval_metrics[k].append(v)
            #             #print(eval_infos['episode'])
            #             final_info['reward'] = eval_infos['episode']['reward']
            #             final_info['return'] = eval_infos['episode']['return']
            #             final_info['episode_len'] = eval_infos['episode']['episode_len']
            #             final_info['success_once'] = eval_infos['episode']['success_once']
            #             final_info['success_at_end'] = eval_infos['episode']['success_at_end']
            #     print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
                # empty = True
                # for k, v in eval_metrics.items():
                #     empty = False
                #     mean = torch.stack(v).float().mean()
                #     # if logger is not None:
                #     #     logger.add_scalar(f"eval/{k}", mean, global_step)
                #     print(f"eval_{k}_mean={mean}")
                # if empty is True:
                #     for k, v in final_info.items():
                #         mean = v.float().mean()
                #         # if logger is not None:
                #         #     logger.add_scalar(f"eval/{k}", mean, global_step)
                #         print(f"eval_{k}_mean={mean}")

            rollout_time = time.time()
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = actor.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
                next_done = torch.logical_or(terminations, truncations).to(torch.float32)
                rewards[step] = reward.view(-1) * args.reward_scale

                if "final_info" in infos:
                    final_info = infos["final_info"]
                    done_mask = infos["_final_info"]
                    with torch.no_grad():
                        final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = actor.get_value(infos["final_observation"][done_mask]).view(-1)

            rollout_time = time.time() - rollout_time
            # bootstrap value according to termination and truncation
            with torch.no_grad():
                next_value = actor.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        next_not_done = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        next_not_done = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                    # next_not_done means nextvalues is computed from the correct next_obs
                    # if next_not_done is 1, final_values is always 0
                    # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                    if args.finite_horizon_gae:
                        """
                        See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                        1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                        lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                        lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                        lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                        We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                        """
                        if t == args.num_steps - 1: # initialize
                            lam_coef_sum = 0.
                            reward_term_sum = 0. # the sum of the second term
                            value_term_sum = 0. # the sum of the third term
                        lam_coef_sum = lam_coef_sum * next_not_done
                        reward_term_sum = reward_term_sum * next_not_done
                        value_term_sum = value_term_sum * next_not_done

                        lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                        reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                        value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                        advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                    else:
                        delta = rewards[t] + args.gamma * real_next_values - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            actor.train()
            b_inds = np.arange(args.ppo_batch_size)
            clipfracs = []
            update_time = time.time()
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.ppo_batch_size, args.ppo_minibatch_size):
                    end = start + args.ppo_minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = actor.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    if args.target_kl is not None and approx_kl > args.target_kl:
                        break

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    ppo_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    ppo_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
                    optimizer.step()

            print("ppo_training done")
            if iteration % args.ppo_log_freq == 0:
                # print(f"Iteration {iteration}, loss: {loss.item()}")
                writer.add_scalar(
                    "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
                )
                # writer.add_scalar("losses/total_loss", loss.item(), iteration)

            if iteration % args.ppo_eval_freq == 0:
                actor.eval()

                def sample_fn(obs):
                    if isinstance(obs, np.ndarray):
                        obs = torch.from_numpy(obs).float().to(device)
                    action = actor.get_action(obs, deterministic=True)
                    if args.sim_backend == "cpu":
                        action = action.cpu().numpy()
                    return action

                with torch.no_grad():
                    eval_metrics = evaluate(args.num_eval_episodes, sample_fn, bc_envs)
                actor.train()
                print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
                for k in eval_metrics.keys():
                    eval_metrics[k] = np.mean(eval_metrics[k])
                    writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                    print(f"{k}: {eval_metrics[k]:.4f}")

                save_on_best_metrics = ["success_once", "success_at_end"]
                for k in save_on_best_metrics:
                    if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                        best_eval_metrics[k] = eval_metrics[k]
                        save_ckpt(run_name, f"best_eval_{k}")
                        print(
                            f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                        )

            if args.save_freq is not None and iteration % args.save_freq == 0:
                save_ckpt(run_name, str(iteration))

        if iteration % args.log_freq == args.log_freq-1:
            # print(f"Iteration {iteration}, loss: {loss.item()}")
            writer.add_scalar(
                "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
            )
            # writer.add_scalar("losses/total_loss", loss.item(), iteration)

        if iteration % args.eval_freq == args.eval_freq-1:
            actor.eval()

            def sample_fn(obs):
                if isinstance(obs, np.ndarray):
                    obs = torch.from_numpy(obs).float().to(device)
                action = actor.get_action(obs, deterministic=True)
                if args.sim_backend == "cpu":
                    action = action.cpu().numpy()
                return action

            with torch.no_grad():
                eval_metrics = evaluate(args.num_eval_episodes, sample_fn, bc_envs)
            actor.train()
            print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
            for k in eval_metrics.keys():
                eval_metrics[k] = np.mean(eval_metrics[k])
                writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
                print(f"{k}: {eval_metrics[k]:.4f}")

            save_on_best_metrics = ["success_once", "success_at_end"]
            for k in save_on_best_metrics:
                if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
                    best_eval_metrics[k] = eval_metrics[k]
                    save_ckpt(run_name, f"best_eval_{k}")
                    print(
                        f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
                    )

        if args.save_freq is not None and iteration % args.save_freq == 0:
            save_ckpt(run_name, str(iteration))


    # for iteration in range(1,args.ppo_num_iterations+1):
    #         final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
    #         actor.eval()

    #         # if iteration % args.ppo_eval_freq == 1:
    #         #     print("Evaluating")
    #         #     eval_obs, _ = eval_envs.reset()
    #         #     eval_metrics = defaultdict(list)
    #         #     num_episodes = 0
    #         #     final_info = {}
    #         #     for _ in range(args.num_eval_steps):
    #         #         with torch.no_grad():
    #         #             eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(actor.get_action(eval_obs, deterministic=True))
    #         #             if "final_info" in eval_infos:
    #         #                 mask = eval_infos["_final_info"]
    #         #                 num_episodes += mask.sum()
    #         #                 for k, v in eval_infos["final_info"]["episode"].items():
    #         #                     eval_metrics[k].append(v)
    #         #             #print(eval_infos['episode'])
    #         #             final_info['reward'] = eval_infos['episode']['reward']
    #         #             final_info['return'] = eval_infos['episode']['return']
    #         #             final_info['episode_len'] = eval_infos['episode']['episode_len']
    #         #             final_info['success_once'] = eval_infos['episode']['success_once']
    #         #             final_info['success_at_end'] = eval_infos['episode']['success_at_end']
    #         #     print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
    #             # empty = True
    #             # for k, v in eval_metrics.items():
    #             #     empty = False
    #             #     mean = torch.stack(v).float().mean()
    #             #     # if logger is not None:
    #             #     #     logger.add_scalar(f"eval/{k}", mean, global_step)
    #             #     print(f"eval_{k}_mean={mean}")
    #             # if empty is True:
    #             #     for k, v in final_info.items():
    #             #         mean = v.float().mean()
    #             #         # if logger is not None:
    #             #         #     logger.add_scalar(f"eval/{k}", mean, global_step)
    #             #         print(f"eval_{k}_mean={mean}")

    #         rollout_time = time.time()
    #         for step in range(0, args.num_steps):
    #             global_step += args.num_envs
    #             obs[step] = next_obs
    #             dones[step] = next_done

    #             # ALGO LOGIC: action logic
    #             with torch.no_grad():
    #                 action, logprob, _, value = actor.get_action_and_value(next_obs)
    #                 values[step] = value.flatten()
    #             actions[step] = action
    #             logprobs[step] = logprob

    #             # TRY NOT TO MODIFY: execute the game and log data.
    #             next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
    #             next_done = torch.logical_or(terminations, truncations).to(torch.float32)
    #             rewards[step] = reward.view(-1) * args.reward_scale

    #             if "final_info" in infos:
    #                 final_info = infos["final_info"]
    #                 done_mask = infos["_final_info"]
    #                 with torch.no_grad():
    #                     final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = actor.get_value(infos["final_observation"][done_mask]).view(-1)

    #         rollout_time = time.time() - rollout_time
    #         # bootstrap value according to termination and truncation
    #         with torch.no_grad():
    #             next_value = actor.get_value(next_obs).reshape(1, -1)
    #             advantages = torch.zeros_like(rewards).to(device)
    #             lastgaelam = 0
    #             for t in reversed(range(args.num_steps)):
    #                 if t == args.num_steps - 1:
    #                     next_not_done = 1.0 - next_done
    #                     nextvalues = next_value
    #                 else:
    #                     next_not_done = 1.0 - dones[t + 1]
    #                     nextvalues = values[t + 1]
    #                 real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
    #                 # next_not_done means nextvalues is computed from the correct next_obs
    #                 # if next_not_done is 1, final_values is always 0
    #                 # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
    #                 if args.finite_horizon_gae:
    #                     """
    #                     See GAE paper equation(16) line 1, we will compute the GAE based on this line only
    #                     1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
    #                     lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
    #                     lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
    #                     lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
    #                     We then normalize it by the sum of the lambda^i (instead of 1-lambda)
    #                     """
    #                     if t == args.num_steps - 1: # initialize
    #                         lam_coef_sum = 0.
    #                         reward_term_sum = 0. # the sum of the second term
    #                         value_term_sum = 0. # the sum of the third term
    #                     lam_coef_sum = lam_coef_sum * next_not_done
    #                     reward_term_sum = reward_term_sum * next_not_done
    #                     value_term_sum = value_term_sum * next_not_done

    #                     lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
    #                     reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
    #                     value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

    #                     advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
    #                 else:
    #                     delta = rewards[t] + args.gamma * real_next_values - values[t]
    #                     advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
    #             returns = advantages + values

    #         # flatten the batch
    #         b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    #         b_logprobs = logprobs.reshape(-1)
    #         b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    #         b_advantages = advantages.reshape(-1)
    #         b_returns = returns.reshape(-1)
    #         b_values = values.reshape(-1)

    #         actor.train()
    #         b_inds = np.arange(args.ppo_batch_size)
    #         clipfracs = []
    #         update_time = time.time()
    #         for epoch in range(args.update_epochs):
    #             np.random.shuffle(b_inds)
    #             for start in range(0, args.ppo_batch_size, args.ppo_minibatch_size):
    #                 end = start + args.ppo_minibatch_size
    #                 mb_inds = b_inds[start:end]

    #                 _, newlogprob, entropy, newvalue = actor.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
    #                 logratio = newlogprob - b_logprobs[mb_inds]
    #                 ratio = logratio.exp()

    #                 with torch.no_grad():
    #                     # calculate approx_kl http://joschu.net/blog/kl-approx.html
    #                     old_approx_kl = (-logratio).mean()
    #                     approx_kl = ((ratio - 1) - logratio).mean()
    #                     clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

    #                 if args.target_kl is not None and approx_kl > args.target_kl:
    #                     break

    #                 mb_advantages = b_advantages[mb_inds]
    #                 if args.norm_adv:
    #                     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    #                 # Policy loss
    #                 pg_loss1 = -mb_advantages * ratio
    #                 pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    #                 pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    #                 # Value loss
    #                 newvalue = newvalue.view(-1)
    #                 if args.clip_vloss:
    #                     v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
    #                     v_clipped = b_values[mb_inds] + torch.clamp(
    #                         newvalue - b_values[mb_inds],
    #                         -args.clip_coef,
    #                         args.clip_coef,
    #                     )
    #                     v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
    #                     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
    #                     v_loss = 0.5 * v_loss_max.mean()
    #                 else:
    #                     v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

    #                 entropy_loss = entropy.mean()
    #                 ppo_loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

    #                 optimizer.zero_grad()
    #                 ppo_loss.backward()
    #                 nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
    #                 optimizer.step()

    #         print("ppo_training done")
    #         if iteration % args.ppo_log_freq == 0:
    #             # print(f"Iteration {iteration}, loss: {loss.item()}")
    #             writer.add_scalar(
    #                 "charts/learning_rate", optimizer.param_groups[0]["lr"], iteration
    #             )
    #             # writer.add_scalar("losses/total_loss", loss.item(), iteration)

    #         if iteration % args.ppo_eval_freq == 0:
    #             actor.eval()

    #             def sample_fn(obs):
    #                 if isinstance(obs, np.ndarray):
    #                     obs = torch.from_numpy(obs).float().to(device)
    #                 action = actor.get_action(obs, deterministic=True)
    #                 if args.sim_backend == "cpu":
    #                     action = action.cpu().numpy()
    #                 return action

    #             with torch.no_grad():
    #                 eval_metrics = evaluate(args.num_eval_episodes, sample_fn, bc_envs)
    #             actor.train()
    #             print(f"Evaluated {len(eval_metrics['success_at_end'])} episodes")
    #             for k in eval_metrics.keys():
    #                 eval_metrics[k] = np.mean(eval_metrics[k])
    #                 writer.add_scalar(f"eval/{k}", eval_metrics[k], iteration)
    #                 print(f"{k}: {eval_metrics[k]:.4f}")

    #             save_on_best_metrics = ["success_once", "success_at_end"]
    #             for k in save_on_best_metrics:
    #                 if k in eval_metrics and eval_metrics[k] > best_eval_metrics[k]:
    #                     best_eval_metrics[k] = eval_metrics[k]
    #                     save_ckpt(run_name, f"best_eval_{k}")
    #                     print(
    #                         f"New best {k}_rate: {eval_metrics[k]:.4f}. Saving checkpoint."
    #                     )

    #         if args.save_freq is not None and iteration % args.save_freq == 0:
    #             save_ckpt(run_name, str(iteration))

    bc_envs.close()
    if args.track:
        wandb.finish()
