# ManiSkill++: Some Extensions for ManiSkill


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
python ./ppo_task.py --env_id=[MobilePutDuckIntoDrawer-v1/MobilePutDuckIntoDrawer-v1/MobilePutDuckIntoDrawer-v1/OpenRefrigerator-v1/MobilePutAppleIntoRefrigerator-v1/MobilePutCubeIntoRefrigerator-v1]
```

```bash
cd ./examples/baselines/sac/
python ./sac_task.py --env_id=[MobilePutDuckIntoDrawer-v1/MobilePutDuckIntoDrawer-v1/MobilePutDuckIntoDrawer-v1/OpenRefrigerator-v1/MobilePutAppleIntoRefrigerator-v1/MobilePutCubeIntoRefrigerator-v1]
```