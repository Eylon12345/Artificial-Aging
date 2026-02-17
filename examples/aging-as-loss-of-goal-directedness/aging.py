from mindcraft import World
from musepy.visualize import Render
from typing import Union
import numpy as np


def rollout_lifetime(world_config,
                     noise_schedule: Union[float, np.ndarray, list, tuple] = None,
                     competency_schedule: Union[float, np.ndarray, list, tuple] = None,
                     ann_schedule: Union[float, np.ndarray, list, tuple] = None,
                     sensitivity_schedule: Union[float, np.ndarray, list, tuple] = None,
                     sensitivity_reduction_rate=0.01,
                     sensitivity_susceptibility=1.0,  # fraction of cells than can lose sensitivity to neighbors
                     memory_reset_on_completion: bool = False,
                     memory_reset_cooldown: int = 0,
                     memory_reset_match_threshold: float = 1.0,
                     max_steps=1000, num_episodes=10, log_fields=("reward",), log_foos=None, verbose=False, render=False,
                     stagnation_cost=False, completion_reward=False,
                     seed=None,
                     ):
    if seed is not None:
        import torch
        torch.manual_seed(seed)
        np.random.seed(seed)

    memory_reset_stats = {"count": 0}
    memory_reset_state = {"last_step": -10**9}

    def set_noise_level(world, noise_level):
        if isinstance(noise_level, (list, tuple)):
            noise_level, cell_idx = noise_level
            world.env.state.noise_idx = cell_idx

        world.env.state.noise_level = noise_level

    def set_competency_level(world, competency_level):
        world.agent.sampling = competency_level

    def update_agent_parameters(world, noise_level):
        parameters = world.agent.get_parameters()
        aged_parameters = parameters + noise_level * np.random.randn(*parameters.shape)
        world.set_parameters(aged_parameters)

    def update_agent_sensitivity(world, sensitivity, sr=sensitivity_reduction_rate, sc=sensitivity_susceptibility):
        world.agent.sensitivity = sensitivity

        # rate at which sensitivity is reduced/gap-junctions (GS) are closed when more GS are open than allowed
        world.agent.sensitivity_reduction_rate = sr
        world.agent.sensitivity_susceptibility = sc

    def reset_memory_if_target_completed(world, enabled):
        if not enabled:
            return

        if memory_reset_cooldown and (world.env.simulation_step - memory_reset_state["last_step"]) <= memory_reset_cooldown:
            return

        state = world.env.task.get_state(world.env)
        target_state = world.env.task.target_state
        match_fraction = np.mean(np.all(state == target_state, axis=-1))
        reached_target = match_fraction >= memory_reset_match_threshold
        if not reached_target:
            return

        policy_module = world.agent.policy_module
        states = getattr(policy_module, "_states", None)
        if states is None:
            return

        for i in range(len(states)):
            states[i][...] = 0.

        memory_reset_stats["count"] += 1
        memory_reset_state["last_step"] = world.env.simulation_step

    schedule = {}
    if noise_schedule is not None:
        schedule[set_noise_level] = noise_schedule

    if competency_schedule is not None:
        schedule[set_competency_level] = competency_schedule

    if ann_schedule is not None:
        schedule[update_agent_parameters] = ann_schedule

    if sensitivity_schedule is not None:
        schedule[update_agent_sensitivity] = sensitivity_schedule

    if memory_reset_on_completion:
        schedule[reset_memory_if_target_completed] = [True for _ in range(max_steps)]

    world_kwargs = dict(max_steps=max_steps,
                        log_fields=log_fields,
                        log_foos=log_foos,
                        schedule=schedule,
                        verbose=verbose,
                        n_episodes=num_episodes,
                        render=bool(render)
                        )

    world = World.make(world_config, **world_kwargs)

    if not stagnation_cost:
        world.env.task.stagnation_cost = 0.

    if not completion_reward:
        world.env.task.completion_reward = 0.

    if render:
        COLORS = ("dodgerblue", "white", "red")
        kwargs = {}
        if isinstance(render, dict):
            kwargs = render
        render_config = dict(colors={"type": COLORS}, ion=True, show=True, palette=COLORS)
        world.env.render = Render.make(render_config, mca=world.env, **kwargs)

    world.max_steps = max_steps
    world.delay = 0.0
    parameters = world.agent.get_parameters().copy()
    for _ in range(num_episodes):
        world.rollout()
        world.set_parameters(parameters)

    data = world.log_history
    data = {k: np.stack([d[k] for d in data], axis=0) for k in log_fields}

    if "reward" in log_fields and world.env.incremental_reward:
        data["cumulative_reward"] = np.cumsum(data["reward"], axis=1)

    if 'state' in log_fields:
        data['type'] = np.argmax(data['state'][..., :3], axis=-1)

    data["noise_schedule"] = noise_schedule
    data["competency_schedule"] = competency_schedule
    data["ann_schedule"] = ann_schedule
    data["sensitivity_schedule"] = sensitivity_schedule
    data["memory_reset_on_completion"] = memory_reset_on_completion
    data["memory_reset_cooldown"] = memory_reset_cooldown
    data["memory_reset_count"] = memory_reset_stats["count"]
    data["memory_reset_match_threshold"] = memory_reset_match_threshold
    return data


def save_data(data_dict, filename="aging.h5"):
    import h5py
    with h5py.File(filename, 'w') as f:
        for k, v in data_dict.items():
            if v is None:
                continue
            f.create_dataset(k, data=v)

    print(f"saved to '{filename}'")


def load_data(filename="aging.h5"):
    import h5py
    dataset = {}
    print(f"loading from '{filename}'")
    with h5py.File(filename, 'r') as f:
        for k in f.keys():
            d = np.array(f.get(k))
            print(f"node: '{k}', shape: {d.shape}")
            dataset[k] = d

    return dataset


def test_noise_schedule(max_steps=100, low_noise=0., high_noise=1., world_config="agents/smiley_16x16/world.yml"):
    noise_schedule = np.linspace(low_noise, high_noise, max_steps)
    rollout_lifetime(world_config=world_config,
                     noise_schedule=noise_schedule, max_steps=max_steps,
                     num_episodes=2,
                     verbose=True, render=True)


def test_competency_schedule(max_steps=100, low_competency=0., max_competency=1., world_config="agents/smiley_16x16/world.yml"):
    noise_schedule = None
    competency_schedule = np.linspace(max_competency, low_competency, max_steps)
    rollout_lifetime(world_config=world_config,
                     noise_schedule=noise_schedule, competency_schedule=competency_schedule,
                     max_steps=max_steps,
                     verbose=True, render=True)


def test_ann_schedule(max_steps=100, ann_noise_level=0.1, world_config="agents/smiley_16x16/world.yml"):
    noise_schedule = None
    competency_schedule = None
    ann_schedule = ann_noise_level
    rollout_lifetime(world_config=world_config,
                     noise_schedule=noise_schedule, competency_schedule=competency_schedule, ann_schedule=ann_schedule,
                     max_steps=max_steps,
                     verbose=True, render=True)


def test_sensor_sensitivity(max_steps=100, num_episodes=5, world_config="agents/smiley_16x16/world.yml"):
    noise_schedule = None
    competency_schedule = None
    ann_schedule = None
    sensitivity_schedule = np.linspace(1.0, 0.0, max_steps)
    rollout_lifetime(world_config=world_config,
                     noise_schedule=noise_schedule, competency_schedule=competency_schedule, ann_schedule=ann_schedule,
                     sensitivity_schedule=sensitivity_schedule,
                     max_steps=max_steps,
                     verbose=True, render=True,
                     num_episodes=num_episodes,
                     log_fields=('reward', 'state'),
                     log_foos={'state': 'observation[:, 0].reshape(16, 16, -1)'}
                     )


def test_sensor_sensitivity_susceptibility(max_steps=100, num_episodes=5, sc=0.3, world_config="agents/smiley_16x16/world.yml"):
    noise_schedule = None
    competency_schedule = None
    ann_schedule = None
    sensitivity_schedule = np.linspace(1.0, 0.0, max_steps)
    rollout_lifetime(world_config=world_config,
                     noise_schedule=noise_schedule, competency_schedule=competency_schedule, ann_schedule=ann_schedule,
                     sensitivity_schedule=sensitivity_schedule, sensitivity_susceptibility=sc,
                     max_steps=max_steps,
                     verbose=True, render=True,
                     num_episodes=num_episodes,
                     log_fields=('reward', 'state'),
                     log_foos={'state': 'observation[:, 0].reshape(16, 16, -1)'}
                     )


def test_memory_reset_on_completion(max_steps=250, num_episodes=5, memory_reset_cooldown=25,
                                    memory_reset_match_threshold=1.0,
                                    world_config="agents/smiley_16x16/world.yml"):
    data = rollout_lifetime(world_config=world_config,
                            max_steps=max_steps,
                            num_episodes=num_episodes,
                            memory_reset_on_completion=True,
                            memory_reset_cooldown=memory_reset_cooldown,
                            memory_reset_match_threshold=memory_reset_match_threshold,
                            log_fields=('reward', 'state'),
                            log_foos={'state': 'observation[:, 0].reshape(16, 16, -1)'},
                            verbose=True,
                            render=True,
                            )
    print(f"memory resets triggered: {data['memory_reset_count']}")
    print(f"memory reset threshold: {data['memory_reset_match_threshold']}")


if __name__ == "__main__":
    import argh
    argh.dispatch_commands([test_noise_schedule,
                            test_competency_schedule,
                            test_ann_schedule,
                            test_sensor_sensitivity,
                            test_sensor_sensitivity_susceptibility,
                            test_memory_reset_on_completion,
                            ])
