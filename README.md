# ManiSkill++: Some Extensions for ManiSkill

## Section 4.1: BC+PPO Finetuning
To get started, we recommend using conda/mamba to create a new environment and install the dependencies

```shell
cd ./examples/baselines/bc/
conda create -n behavior-cloning-ms python=3.9
conda activate behavior-cloning-ms
pip install -e .
```

First download and replay the demonstrations:

```bash
python -m mani_skill.utils.download_demo "PullCube-v1"
```

```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ~/.maniskill/demos/PullCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o state \
  --save-traj --num-procs 10 -b cpu
```

Then run the training script:

```bash
cd ./examples/baselines/bc/
python bc_with_ppo.py --env-id "PullCube-v1" --demo-path ~/.maniskill/demos/PullCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.cpu.h5 --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --max-episode-steps 100 --total-iters 10000 --num_envs=2048 --update_epochs=8 --num_minibatches=32 --num-steps=100 --num-eval-steps=100
```

## Section 4.2: PPO+Distillation

```bash
cd ./examples/baselines/ppo/
python ./ppo_mixed.py --model_type=teacher --env_id=[PlaceSphere-v1/PullCube-v1/LiftPegUpright-v1/PokeCube-v1]
python ./ppo_mixed.py --model_type=student --teacher_path=[teacher_checkpoint_path] --env_id=[PlaceSphere-v1/PullCube-v1/LiftPegUpright-v1/PokeCube-v1]
```

## Section 5: Our Designed Tasks

Put data in the ``data`` folder into ``~/``
```bash
cd ./examples/baselines/ppo/
python ./ppo_task.py --env_id=[MobilePutCubeIntoDrawer-v1/MobilePutDuckIntoDrawer-v1/MobilePutPyramidIntoDrawer-v1/OpenRefrigerator-v1/MobilePutAppleIntoRefrigerator-v1/MobilePutCubeIntoRefrigerator-v1]
```

```bash
cd ./examples/baselines/sac/
python ./sac_task.py --env_id=[MobilePutCubeIntoDrawer-v1/MobilePutDuckIntoDrawer-v1/MobilePutPyramidIntoDrawer-v1/OpenRefrigerator-v1/MobilePutAppleIntoRefrigerator-v1/MobilePutCubeIntoRefrigerator-v1]
```