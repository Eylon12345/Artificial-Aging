""" Experimentation utilities for Neural Cellular Automata (NCA) models based on [1]

[1] B. Hartl, S. Risi, M. Levin, "Evolutionary Implications of Self-Assembling Cybernetic Materials with Collective Problem-Solving Intelligence at Multiple Scales", Entropy 26(7), 532, 2024; https://doi.org/10.3390/e26070532
"""

import json
import yaml
import numpy as np
import os
# import matplotlib.pyplot as plt
from matplotlib import colors
# import matplotlib.colors as mcolors
# from matplotlib.patches import Rectangle
from mindcraft.util.hpc import get_dst_path

COLORS = ("black", "dodgerblue", "red", "lightgreen", "yellow", "grey", "magenta", "orange", "lightblue", "brown")
COLORAMA = ("BLACK", "LIGHTBLUE_EX", "RED", "LIGHTGREEN_EX", "LIGHTYELLOW_EX", "WHITE", "MAGENTA", "YELLOW", "CYAN", "LIGHTRED_EX")
RGBA = tuple((colors.to_rgba(c) for c in COLORS))
CHANNELS = [i for i in range(len(COLORS))]

DEFAULT_WORLD_NAME = "world.yml"
DEFAULT_AGENT_NAME = "agent.yml"
DEFAULT_ENV_NAME = "env.yml"


def get_agent_file(path, agent_name=DEFAULT_AGENT_NAME):
    return os.path.join(path, agent_name)


def get_env_file(path, env_name=DEFAULT_ENV_NAME):
    return os.path.join(path, env_name)


def get_world_file(path, world_name=DEFAULT_WORLD_NAME):
    return os.path.join(path, world_name)


def load_dataset(folder="tasks"):
    """ Load a tasks dataset from a folder containing json files. """
    import os
    import json

    json_files = [pos_json for pos_json in os.listdir(folder) if pos_json.endswith('.json')]
    data = {}
    for js in json_files:
        with open(os.path.join(folder, js)) as json_file:
            data[js] = json.load(json_file)
    return data


def load_config(filename):
    if filename.endswith(".yml"):
        with open(filename, 'r') as f:
            import yaml
            config = yaml.safe_load(f)

    elif filename.endswith(".json"):
        with open(filename, 'r') as f:
            import json
            config = json.load(f)

    return config


def to_muse_env(dataset, task_id=None, mode=None, init_target=False,
                state_config=None, grid_config=None, render_config=None, task_config=None,
                nca_config="HybridNCAAgent",  # or "GeneNCAAgent"
                ):
    """ Transform a task within the dataset into `HybridNCA` multicellular environment.

    :return: `HybridNCA` environment.
    """
    from musepy.util.geometric import SquareGrid as Grid
    grid = Grid.make(grid_config)

    from musepy.util.state import State
    state = State.make(state_config)

    from musepy.corpus import Task
    task_config["corpus"] = dataset[task_id]
    task_config["mode"] = mode
    task_config["init_target"] = init_target
    task = Task.make(task_config)

    from musepy.visualize import Render
    render = Render.make(render_config)

    from musepy.envs import HybridNCA
    nca_env = HybridNCA(grid=grid, state=state, task=task, render=render)

    return nca_env


def get_wildcard_agent(agent_file, env, solver=None, **kws):
    """ Load an agent file and replace wildcards with environment specific values. """
    from mindcraft import Agent

    try:
        return Agent.make(agent_file)

    except TypeError:
        with open(agent_file, "r") as f:
            agent = yaml.safe_load(f)

            eval_keys = []
            def recursive_replace(d: dict, wildcard: str, value: object):
                for k, v in d.items():
                    if isinstance(v, dict):
                        recursive_replace(v, wildcard, value)

                    elif v == wildcard:
                        d[k] = value

                    elif isinstance(v, str) and wildcard in v:
                        if not any((d == di and k == ki) for di, ki in eval_keys):
                            eval_keys.append((d, k))
                        d[k] = v.replace(wildcard, str(value))

            recursive_replace(agent, wildcard="{ACTION_SPACE}", value=f"Box(-1, 1, ({env.n_channels},), dtype=np.float32)")
            recursive_replace(agent, wildcard="{INPUT_SIZE}", value=env.n_channels)
            recursive_replace(agent, wildcard="{OUTPUT_SIZE}", value=env.n_channels)
            recursive_replace(agent, wildcard="{NEIGHBORHOOD}", value=env.grid.neighborhood_size)
            recursive_replace(agent, wildcard="{NUM_CELLS}", value=env.n_cells)
            recursive_replace(agent, wildcard="{NUM_CHANNELS}", value=env.n_channels)

            if solver is not None:
                if "num_letters" in solver.get("opts", {}):
                    recursive_replace(agent, wildcard="{NUM_LETTERS}", value=solver["opts"]["num_letters"])

            for d, k in eval_keys:
                # compile math expression
                d[k] = eval(d[k])

            for k, v in kws.items():
                agent[k] = v

            agent = Agent.make(agent)

    return agent


def train(task_id,
          agent,
          config,
          tasks=None,
          path="data/model/",
          new_model=False,
          checkpoint=None,
          env_name=DEFAULT_ENV_NAME,
          agent_name=DEFAULT_AGENT_NAME,
          world_name=DEFAULT_WORLD_NAME,
          ):
    """ Train an `musepy.agents.HybridNCAAgent` agent on a morphogenesis task.

    :param task_id: specific id for a training corpus (i.e., a filename within the `tasks` folder).
    :param agent: A path to an agent file - the file-name will be used to generate a path-tree specifying the
                  task, the agent and the world configuration.
    :param config: A path to a configuration file.
    :param tasks: A path to a folder containing json files with training corpora.
    :param path: A path to store the model.
    :param new_model: [Optional] start training from scratch.
    :param checkpoint: [Optional] continue training from a checkpoint (use `True` for latest or specify as dictionary `{runs: int, gens: int}`; -1 will load the latest).
    :param env_name: [Optional] name of the environment file in the destination folder.
    :param agent_name: [Optional] name of the agent file in the destination folder.
    :param world_name: [Optional] name of the world file in the destination folder.

    :return: training history.
    """

    env_path = get_env_file(path, env_name=env_name)
    world_path = get_world_file(path, world_name=world_name)
    agent_path = get_agent_file(path, agent_name=agent_name)

    from mpi4py import MPI
    task_config = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        kwargs = {}
        if tasks is not None:
            kwargs["folder"] = tasks

        dataset = load_dataset(**kwargs)

        print(f"Loading config '{config}'")
        config = load_config(config)

        print(f"Parsing task '{task_id}'")
        grid_config = config['grid_config']
        state_config = config['state_config']
        task_config = config['task_config']
        render_config = config['render_config']
        agent_config = config.get("agent_config", {})
        nca_config = config.get("nca_config", "HybridNCAAgent")
        muse_env = to_muse_env(dataset, task_id=task_id, mode="train", init_target=False,
                               grid_config=grid_config, state_config=state_config, task_config=task_config,
                               render_config=render_config, nca_config=nca_config)

        assert np.array_equal(muse_env.task.start_state.shape, muse_env.task.target_state.shape)
        print(f"Dumping Environment to '{env_path}'")
        muse_env.to_yml(filename=env_path)

        if new_model or not os.path.exists(agent_path):
            if agent is None:
                raise ValueError("Please provide an `agent` representation.")

            if not agent.endswith(".yml"):
                agent += ".yml"

            from mindcraft import Agent
            if new_model:
                print(f"Loading agent '{agent}'")
                agent = get_wildcard_agent(agent, muse_env, solver=config['solver_config'], **agent_config)
                print(f"Dumping Agent to '{agent_path}'")
                agent.to_yml(filename=agent_path)

        print(f"Dumping world to '{world_path}'")
        with open(world_path, 'w') as s:
            from yaml import safe_dump as yml_safe_dump
            world = config['world_config']
            world['agent'] = agent_path  # path to agent file
            world['env'] = env_path  # path to environment (parsed task_id from database)
            world['verbose'] = False  # io helpers, usually turned off for training
            world['render'] = False
            yml_safe_dump(world, stream=s, default_flow_style=False, sort_keys=False)

        if 'file_name' not in config['solver_config']:
            config['solver_config']['file_name'] = agent_name

    # start training
    from mindcraft.script import train as train_agent
    config = MPI.COMM_WORLD.bcast(config, root=0)
    if isinstance(checkpoint, str):
        checkpoint = json.loads(checkpoint)

    return train_agent(world_repr=world_path, path=path, checkpoint=checkpoint, **config['solver_config'])


def test(task_id,
         mode="test",
         task_num=0,
         path="data/model/",
         world_config=None,
         schedule="random",
         render_config=None,
         disable_render=False,
         tasks=None,
         world_name=DEFAULT_WORLD_NAME,
         agent_name="",
         checkpoint="{}",
         quiet=False,
         max_steps=50,
         n_episodes=5,
         delay=0.05,
         log_fields=(),
         log_foos=(),
         ):
    """ Test an agent on a morphogenesis task.

    :returns: log history, i.e., a list of dictionaries (one dict per episode),
              each containing the log fields across the time-steps of the episode.
    """
    from mindcraft import World
    from musepy.visualize import Render
    from musepy.util.state import State

    agent_path = get_agent_file(path, agent_name=agent_name)
    world_path = get_world_file(path, world_name=world_name)
    print(f"Loading world from '{world_path}'")

    if not disable_render:
        if isinstance(render_config, str):
            render_config = load_config(render_config)

        if render_config is None:
            render_config = dict(colors={State.TYPE_CHL: COLORS}, ion=True, show=True, palette=COLORS)
            # render_config = dict(colors={SC.HIDDEN_CHL: "viridis"}, ion=True, show=True, palette=None)
            # render_config = dict(colors={SC.TYPE_CHL: "magma"}, ion=True, show=True, palette=COLORS, single_channel=1)

    world_config = world_config or {"verbose": not quiet, "render": not disable_render, "delay": delay,
                                    "n_episodes": n_episodes, "max_steps": max_steps,
                                    "log_fields": log_fields, "log_foos": log_foos or {},
                                    }
    if isinstance(world_config, str):
        import json
        world_config = json.loads(world_config)

    if agent_name:
        with open(world_path, 'r') as s:
            world_loaded = yaml.safe_load(s)

        agent_path = get_agent_file(path, agent_name=agent_name)
        world_loaded["agent"] = agent_path
        world = World.make(world_loaded, **world_config)

    else:
        world = World.make(world_path, **world_config)
        agent_path = get_agent_file(path, agent_name=DEFAULT_AGENT_NAME)

    if checkpoint:
        if isinstance(checkpoint, str):
            import json
            checkpoint = json.loads(checkpoint)

        from mindcraft.train import EvolutionaryStrategy
        log_file = agent_path.replace(".yml", ".log")
        agent_parameters = EvolutionaryStrategy.load_checkpoint(log_file, **checkpoint)
        world.agent.set_parameters(agent_parameters)

    if tasks is not None:
        data = load_dataset(folder=tasks)
        world.env.task.corpus = data[task_id]

    world.env.task.mode = mode
    world.env.task.task_num = task_num
    world.env.task.schedule = schedule

    if render_config is not None and not disable_render:
        world.env.render = Render.make(render_config, mca=world.env)
    else:
        world.render = False
        world.env.render = Render()

    for n in range(world.n_episodes):
        world.rollout()
        world.reset()

    # remove the first log entry (initial state)
    log_history = [{k: v[1:] for k, v in h.items()} for h in world.log_history]
    return log_history  # returns a list of dictionaries (one dict per episode), each containing the log fields


def progress(log_file, xlim=None, file_title=False):
    """ Plot the training progress of an agent. """
    import numpy as np
    import matplotlib.pyplot as plt
    from mindcraft.train import EvolutionaryStrategy
    import os

    if not os.path.exists(log_file):
        raise FileNotFoundError(log_file)

    runs = EvolutionaryStrategy.load_log(log_file, to_pandas=True)
    print(f"Accessing Log-File `{log_file}`")
    print(f"History: {len(runs)} runs, {len([r for r in runs if r['tail'] != []])} of them finished")

    p_start = 0
    best = []
    cost = []
    mean = []
    std = []
    steps = []
    boosts = []
    for i in range(len(runs)):
        data = runs[i]['data']
        if not len(data):
            continue

        best.append(data['best'])
        cost.append(data['cost'])
        mean.append(data['mean'])
        std.append(data['std'])
        steps.append(data['step'] + p_start)

        boosts.append(max(data['step']))
        p_start += boosts[-1] + 1

    best = np.concatenate(best)
    cost = np.concatenate(cost)
    mean = np.concatenate(mean)
    std = np.concatenate(std)
    steps = np.concatenate(steps)

    plt.figure(figsize=(15, 5))
    ax = plt.gca()

    # plot_foo(steps, mean, label='mean')
    plt.plot(steps, best, label='Hist. Fittest', zorder=2, linewidth=2)
    plt.fill_between(steps, cost, best, label='Current Fittest', zorder=3, alpha=0.5)
    plt.plot(steps, mean, color="black", zorder=1, label="Pop. Mean", linewidth=2)
    plt.fill_between(steps, mean-std, np.minimum(best, mean+std), zorder=3, alpha=0.5, color="gray", label="Pop. Std.")

    if xlim is not None:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim([1, max(steps)])

    d_min = min([min(mean), min(cost)])
    d_max = max([max(best), max(cost)])
    offset = 0.05 * (d_max - d_min)

    # for b in np.cumsum(boosts):
    #     plot_foo([b, b], [d_min - offset, d_max + offset], 'k--')
    ax.set_ylim([d_min - offset, d_max + offset])

    ax.set_xlabel('Generations')
    ax.set_ylabel('Return (Cumulative Reward)')

    if file_title:
        plt.title(log_file)
    plt.grid()
    plt.legend()
    plt.show()


def load_genomes(checkpoint_file, run_id=-1, normalize=True):
    """ Load genomes from a checkpoint file. """
    import h5py
    with h5py.File(checkpoint_file, 'r') as h5:
        elite_param = []
        pop_param = []
        chkpts = []
        elite_reward = []
        pop_reward = []
        mean_reward = []
        std_reward = []
        for run in list(sorted([int(k) for k in h5.keys()])):
            lx = h5[str(run)]['x']

            chkpts_r = sorted([int(k) for k in lx.keys()])  # EA-cycles
            lx = np.asarray([lx[str(k)] for k in chkpts_r])

            offset = 0
            if len(chkpts) and run_id is None:
                offset = chkpts[-1][-1]
            chkpts_r = [i + offset for i in chkpts_r]

            r, p = h5[str(run)]['reward'], h5[str(run)]['parameters']
            r_keys = [int(ri) for ri in r.keys()]
            gen_order = [str(rj) for rj in sorted(r_keys)]
            rx = [np.max(r[gen]) for gen in gen_order]
            rm = [np.mean(r[gen]) for gen in gen_order]
            rs = [np.std(r[gen]) for gen in gen_order]
            pp = [p[gen][()] for gen in gen_order]
            rp = [r[gen][()] for gen in gen_order]

            elite_param.append(lx)
            elite_reward.append(rx)
            pop_param.append(pp)
            pop_reward.append(rp)
            chkpts.append(chkpts_r)
            mean_reward.append(rm)
            std_reward.append(rs)

    if run_id is None:
        elite_param = np.concatenate(elite_param)
        elite_reward = np.concatenate(elite_reward)
        pop_param = np.concatenate(pop_param)
        pop_reward = np.concatenate(pop_reward)
        chkpts = np.concatenate(chkpts)
        mean_reward = np.concatenate(mean_reward)
        std_reward = np.concatenate(std_reward)

    else:
        elite_param = np.array(elite_param[run_id])
        elite_reward = np.array(elite_reward[run_id])
        pop_param = np.array(pop_param[run_id])
        pop_reward = np.array(pop_reward[run_id])
        chkpts = np.array(chkpts[run_id])
        mean_reward = np.array(mean_reward[run_id])
        std_reward = np.array(std_reward[run_id])

    if not normalize:
        return chkpts, elite_param, elite_reward, pop_param, pop_reward, mean_reward, std_reward,

    # scale (to show changes in parameters) and return
    pp = elite_param - np.mean(elite_param, axis=0)
    pp /= np.mean(elite_param, axis=0)[None, :]
    pop_param = pop_param - np.mean(elite_param, axis=0)
    pop_param /= np.mean(elite_param, axis=0)[None, :]
    return chkpts, pp, elite_reward, pop_param, pop_reward, mean_reward, std_reward


def checkpoints(task_id,
                mode="test",
                task_num=0,
                path="data/model/",
                schedule="random",
                tasks=None,
                world_name=DEFAULT_WORLD_NAME,
                verbose=False,
                n_episodes=5,
                run=-1,
                show_progress=False,
                show_states="palette",
                elites=False,
                noise_level=None,
                ):
    """ Evaluate structural and genotypic fitness of a population of agents.

    :param task_id: chose e.g., "czech_8x8.json", "french_6x6.json", ... from the `tasks` folder.
    :param mode: [Optional] chose from "test" or "train".
    :param task_num: [Optional] chose the task number (TODO).
    :param path: [Optional] path to load the results from.
    :param schedule: [Optional] chose from "random" or "sequential" (TODO).
    :param tasks: [Optional] folder containing the tasks.
    :param world_name: [Optional] name of the world file in the destination folder.
    :param verbose: [Optional] print additional information during evaluation.
    :param n_episodes: [Optional] number of episodes to evaluate the agent.
    :param run: [Optional] run id of the checkpoint to evaluate.
    :param show_progress: [Optional] show the progress of the agent.
    :param show_states: [Optional] show the states of the agent (choose from "palette" or "rgb").
    :param elites: [Optional] evaluate only the elite genomes (historically best), or the best genomes of each generation.
    :param noise_level: [Optional] set the noise level of the environment.

    :return: data dictionary containing structural and genotypic fitness, states, and reward statistics.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mindcraft.agents import TorchAgent
    from mindcraft import World

    world_path = get_world_file(path, world_name=world_name)
    world_config = load_config(world_path)
    agent_path = world_config['agent']

    chkpt_file = agent_path.replace(".yml", "") + "-ckpt.h5"
    print(f"Loading checkpoint from '{chkpt_file}'")
    chkpts, genome, reward, pop_genomes, pop_rewards, mean_reward, std_reward = load_genomes(chkpt_file, run_id=run, normalize=False)
    print("Found {0} checkpoint generations of run {1}".format(len(chkpts), run))

    if not elites:
        genome = [p[np.argmax(r)] for p, r in zip(pop_genomes, pop_rewards)]
        reward = [np.max(r) for r in pop_rewards]

    # LOAD WORLD
    agent = TorchAgent.make(agent_path)
    genome_names = agent.unfreeze
    world_config = {"verbose": verbose, "render": False, "delay": 0.0, "n_episodes": n_episodes}
    world = World.make(world_path, **world_config)

    if tasks is not None:
        data = load_dataset(folder=tasks)
        world.env.task.corpus = data[task_id]

    world.env.task.mode = mode
    world.env.task.task_num = task_num
    world.env.task.schedule = schedule

    # EVALUATE STRUCTURAL GENOME AND FITNESS
    genotypic_reward = []
    structural_reward = []
    structural_reward_std = []
    structural_reward_max = []

    # disable actions by NCA's cells, only structural genome is evaluated
    sampling, world.agent.sampling = world.agent.sampling, 0.         # this is the sampling rate / decision-making probability from [1]
    sampling_scale, world.agent.sampling_scale = world.agent.sampling_scale, -1.

    # GENOME FITNESS: EVALUATE TASK ON STRUCTURAL GENOME WITHOUT DEVELOPMENT
    world.n_episodes = 1
    world.env.state.noise_level = noise_level if noise_level is not None else world.env.state.noise_level
    max_steps, world.max_steps = world.max_steps, 0
    for i, p in enumerate(genome):
        str_reward = World.make_rollouts(world, p, verbose=False)
        genotypic_reward.append(np.mean(str_reward))
        world.env.reset()
        print("\rEvaluating structural genome {0}/{1}".format(i + 1, len(genome)), end="")
    print("\rEvaluating structural genome {0}/{1}".format(len(genome), len(genome)))

    # STRUCTURAL FITNESS: EVALUATE STRUCTURE DURING DEVELOPMENT WITHOUT CELLULAR DECISION-MAKING (ONLY NOISE)
    world.n_episodes = n_episodes
    world.max_steps = max_steps
    for i, p in enumerate(genome):
        str_reward = World.make_rollouts(world, p, verbose=False)
        structural_reward.append(np.mean(str_reward))
        structural_reward_std.append(np.std(str_reward))
        structural_reward_max.append(np.max(str_reward))
        world.env.reset()
        print("\rEvaluating structural reward {0}/{1}".format(i + 1, len(genome)), end="")
    print("\rEvaluating structural reward {0}/{1}".format(len(genome), len(genome)))

    # STATES: EVALUATE DEVELOPMENTAL STATES WITH NCAs DECISION-MAKING
    world.agent.sampling = sampling
    world.agent.sampling_scale = sampling_scale
    world.n_episodes = 1
    world.log_fields = ["state"]
    shape = world.env.shape
    world.log_foos = {"state": f"self.env.state_array.copy()"}
    states = []
    for i, p in enumerate(genome):
        World.make_rollouts(world, p, verbose=False)
        state = np.array([h["state"][0] for h in world.log_history[-max_steps:]])
        states.append(state)
        world.env.reset()
        world.log_history = []
        print("\rEvaluating states {0}/{1}".format(i + 1, len(genome)), end="")
    print("\rEvaluating states {0}/{1}".format(len(genome), len(genome)))

    data = dict(
        generations=chkpts,
        genome=genome,
        reward=reward,
        mean_reward=mean_reward,
        std_reward=std_reward,
        structural_genome=genotypic_reward,
        structural_reward=structural_reward,
        states=states,  # generations x developmental_steps x shape
    )

    if show_progress:
        steps = chkpts

        plt.figure(figsize=(15, 5))
        ax = plt.gca()

        best = [reward[0]]
        for i, r in enumerate(reward[1:]):
            best.append(max(r, best[-1]))

        # plot_foo(steps, mean, label='mean')
        plt.plot(steps, best, label='Hist. Fittest', zorder=2, linewidth=2)
        plt.fill_between(steps, reward, best, label='Current Fittest', zorder=3, alpha=0.5)
        plt.plot(steps, mean_reward, color="black", zorder=1, label="Pop. Mean", linewidth=2)
        plt.fill_between(steps, mean_reward - std_reward, np.minimum(best, mean_reward + std_reward), zorder=3, alpha=0.5, color="gray", label="Pop. Std.")

        # plot structural fitness
        ax.plot(steps, structural_reward, label='Structural Fitness', zorder=2, linewidth=2, color="purple", alpha=0.8)
        ax.fill_between(steps, np.array(structural_reward) - np.array(structural_reward_std), structural_reward_max, color="purple", alpha=0.2)
        ax.plot(steps, genotypic_reward, label='Genotypic Fitness', zorder=2, linewidth=2, color="magenta", alpha=0.8)

        ax.set_xlabel('Generations')
        ax.set_ylabel('Return (Cumulative Reward)')

        plt.grid()
        plt.legend()

    if show_states in ("rgb", "palette"):
        interval = max([1, len(states) // world.max_steps * 2])
        offset, scale = world.env.state.bounds[0], world.env.state.bounds[1] - world.env.state.bounds[0]
        print("Plotting {} generations of {} developmental states".format(len(states), len(states[0])), end="")
        fig, axes = plt.subplots(world.max_steps, len(states) // interval, figsize=(12, 12), sharex=True, sharey=True)
        for i, (gen, state, ax_col) in enumerate(zip(chkpts[::interval], states[::interval], axes.T)):
            print("\rPlotting {0} generations of {1} developmental states: {2}/{3}".format(len(states), len(states[0]), i+1, len(states) // interval), end="")
            for k, ax in enumerate(ax_col):
                if show_states == "rgb":
                    ax.imshow((state[k, :, :, :3].transpose(1, 0, 2) - offset) / scale)
                else:
                    # use 3 states and plot with color-names specified in world.env.task.palette
                    argmax_state = np.argmax(state[k, :, :, :3], axis=-1).T
                    ax.imshow(argmax_state, cmap=colors.ListedColormap(world.env.task.palette), vmin=0, vmax=2)

                ax.invert_yaxis()
                ax.set_xticks([])
                ax.set_yticks([])
                if not i:
                    ax.set_ylabel("{0}".format(k))
                    ax_col[-1].set_ylabel("Dev. {0}".format(k))
                    ax_col[-1].set_xlabel("Gen. {0}".format(gen))
                else:
                    ax_col[-1].set_xlabel("{0}".format(gen))

        print("\rPlotting {0} generations of {1} developmental states: {2}/{3}".format(len(states), len(states[0]), len(states) // interval, len(states) // interval))
        plt.tight_layout()

    if show_states or show_progress:
        plt.show()

    return data




if __name__ == '__main__':
    import argh
    argh.dispatch_commands([train,
                            test,
                            progress,
                            ])
