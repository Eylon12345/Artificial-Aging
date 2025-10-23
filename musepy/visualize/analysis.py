import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mindcraft.train import EvolutionaryStrategy as ES
from mindcraft import World

FILE_PATH_MASK = "{data_path}/{env}/noise_{noise}-es_{es}/sm_{sm}-embd_{embd}-pm_{pm}-rdcy_{rdcy}-hook_{hook}-state_{state}-grid_{grid}-fwd_{fwd}"
LOG_FIELDS = ('reward', 'state', 'embedding', 'action')


def get_files(data_path,
              file_mask=FILE_PATH_MASK,
              **kwargs):

    file_path = file_mask.format(data_path=data_path, **{k: str(v) for k, v in kwargs.items() if k in file_mask})
    print("Opening", file_path)

    world_file = os.path.join(file_path, "world.yml")
    agent_file = os.path.join(file_path, "agent.yml")
    log_file = os.path.join(file_path, "agent.log")

    return world_file, agent_file, log_file


def load_data(data_path,
              file_mask=FILE_PATH_MASK,
              run=-1,
              delta_gen=1,
              keys=("x", "reward"),
              **kwargs,
              ):
    world_file, agent_file, log_file = get_files(data_path=data_path, file_mask=file_mask, **kwargs)

    chkpt_file = ES.get_checkpoint_file(log_file)
    with h5py.File(chkpt_file, "r") as f:
        runs = list(sorted([int(run) for run in f.keys()]))

        run = runs[run]
        run_key = str(run)
        run_node = f[run_key]

        reward_node = run_node['reward']

        generations = list(sorted([int(k) for k in reward_node.keys()]))
        population_rewards = [reward_node[str(g)][()] for g in generations]
        elite_reward = np.max(population_rewards, axis=1)

        data = []
        for gen, reward, population_reward in zip(generations, elite_reward, population_rewards):
            if len(data) and "reward" in data[-1] and reward <= data[-1]["reward"]:
                continue

            if len(data) and gen - data[-1]["gen"] < delta_gen and reward != max(elite_reward) and delta_gen >= 1:
                continue

            gen_i = np.argmin(np.abs(np.array(generations) - gen))
            gen_key = str(generations[gen_i])

            # extract parameters
            gen_data = {"gen": int(gen_key)}
            for k in keys:
                if k == "reward":
                    gen_data[k] = reward
                elif k == "population_reward":
                    gen_data[k] = population_reward
                else:
                    gen_data[k] = run_node[k][gen_key][()]

            data.append(gen_data)

    return world_file, data


def get_trajectory(data_path,
                   file_mask=FILE_PATH_MASK,
                   env="czech_8x8", noise="0.25", es="CMAES",
                   sm="ff", embd="mean", pm="ff",
                   rdcy=1, hook="none",
                   state="T3H1",
                   grid="sqr",
                   fwd="0.5",
                   run=-1,
                   n_episodes=10,
                   delta_gen=25,
                   max_dev_t=26,
                   threshold=64,
                   show_type=False,
                   quite=False,
                   N_embedding=8,
                   N_act=4,
                   ):

    world_file, data = load_data(data_path=data_path, file_mask=file_mask,
                                 env=env, noise=noise, es=es,
                                 sm=sm, embd=embd, pm=pm,
                                 rdcy=rdcy, hook=hook,
                                 state=state,
                                 grid=grid,
                                 fwd=fwd,
                                 run=run,
                                 delta_gen=delta_gen,
                                 keys=("x", "reward"),
                                 )

    Nx = env.split("_")[1].split("x")[0]
    Ny = env.split("_")[1].split("x")[1]
    N_embd = N_embedding
    N_act = N_act
    LOG_FOOS = {'state': f'observation[:, 0].reshape({Nx}, {Ny}, -1)',
                'embedding': f'self.agent.embedding.detach().numpy().reshape({Nx}, {Ny}, -1) if self.agent.embedding is not None else np.zeros(({Nx}, {Ny}, {N_embd}))',
                'action': f'action.reshape({Nx}, {Ny}, -1) if not len(action) == {N_act} else np.zeros(({Nx}, {Ny}, {N_act}))',
                }

    for d in data:
        print(f"Evaluate generation {d['gen']}")
        world = World.make(world_file, verbose=not quite, log_fields=LOG_FIELDS, log_foos=LOG_FOOS)
        world.agent.set_parameters(d["x"])
        world.n_episodes = n_episodes
        if max_dev_t == 0:
            max_dev_t = world.max_steps

        states, rewards, geno_fitness = [], [], []
        for _ in range(n_episodes):
            world.rollout()
            states.append(world.log_history[-1]["state"][1:])
            rewards.append(world.log_history[-1]["reward"][1:])
            geno_fitness.append(world.log_history[-1]["reward"][1])

        best_episode = np.argmax(np.array(rewards)[:, (max_dev_t -1) if max_dev_t else -1], axis=0)

        if show_type:
            d["state"] = np.argmax(states[best_episode][..., :3], axis=-1)
        else:
            d["state"] = (states[best_episode][..., :3] + 3) / 6
            # change red and blue
            red = d["state"][..., 0].copy()
            d["state"][..., 0] = d["state"][..., 2]
            d["state"][..., 2] = red

        d["fitness"] = np.mean(rewards, axis=0)
        d["geno_fitness"] = np.mean(geno_fitness)

    n, m = max_dev_t, len(data)
    gs = GridSpec(1 + n + 1 + 1,  # rows: fitness, states, blank, arrow
                  1 + 1 + m,      # cols: arrow, blank, states
                  width_ratios=[0.025 * m] * 2 + [1] * m, height_ratios=[0.125 * n] + [1] * n + [0.025 * n] * 2)
    f = plt.figure(figsize=(12, 12))

    # plot reward for selected generations
    ax_reward = f.add_subplot(gs[0, 2:])
    rewards = np.array([d["reward"] for d in data])
    ax_reward.plot(rewards, marker="o", color="cornflowerblue", zorder=2, label="Current Fitness")
    ax_reward.plot([d["geno_fitness"] for d in data], color="purple", zorder=2, label="Structural Fitness", marker="+")
    ax_reward.legend()
    ax_reward.set_ylabel("Fitness")
    # draw horizontal line at 64 threshold
    ylim = min(rewards) - (max(rewards) - min(rewards)) * 0.2, max(rewards) + (max(rewards) - min(rewards)) * 0.1
    ax_reward.vlines(np.where(rewards >= threshold)[0][0], *ylim, color="green", zorder=1, linestyles="--", linewidth=3)
    ax_reward.set_yticks([0, threshold])
    ax_reward.set_ylim(*ylim)
    # set x labels to specific generation list
    ax_reward.set_xticks(range(len([d["gen"] for d in data])))
    ax_reward.set_xticklabels([])
    ax_reward.set_xlim(-0.5, len(data) - 0.5)
    # rotate x labels by 90°
    for tick in ax_reward.get_xticklabels():
        tick.set_rotation(90)
    # show grid
    ax_reward.grid(which='both', zorder=0)

    # plot state over developmental time (vertical) for selected generations (horizontal)
    for t in range(max_dev_t):
        for i, d in enumerate(data):
            ax = f.add_subplot(gs[1 + t, 2 + i])
            if not show_type:
                ax.imshow(d["state"][t].transpose(1, 0, 2), interpolation='none')  # , cmap=cmap, interpolation='none')
            else:
                cmap = plt.cm.colors.ListedColormap(['cornflowerblue', 'red', 'white'])
                ax.imshow(d["state"][t].T, interpolation='none', cmap=cmap)

            ax.set_xticks([])
            ax.set_yticks([])

            if t == max_dev_t - 1:
                ax.set_xlabel(str(d['gen']), rotation="90")

            if i == 0 and not t % 5:
                ax.set_ylabel(str(t))

    # draw horizontal arrow in the last row
    ax_foot = f.add_subplot(gs[-1, 1:])
    ax_foot.axis('off')  # Hide the axes
    ax_foot.arrow(0.5, 0.5, 1.0, 0, head_width=0.01, head_length=0.025, fc='k', ec='k')
    ax_foot.text(0.5, -0.25, "Evolution (Generations)", ha="center", va="center", rotation=0,
                 # bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'),
                 transform=ax_foot.transAxes)  # Use transAxes to specify the coordinate system

    # draw horizontal arrow in the last row
    ax_left = f.add_subplot(gs[1:-1, 0])
    ax_left.axis('off')  # Hide the axes
    ax_left.arrow(0.5, 0.5, 0., -1.0, head_width=0.01, head_length=0.025, fc='k', ec='k')
    ax_left.text(-0.25, 0.5, "Morphogenesis (Developmental Steps)", ha="center", va="center", rotation=90,
                 # bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'),
                 transform=ax_left.transAxes)  # Use transAxes to specify the coordinate system

    multiscale_file = world_file.replace("world.yml", "") + f"multiscale_morphogenesis-state_{'RGB' if not show_type else 'TYPE'}.pdf"
    plt.savefig(multiscale_file, bbox_inches='tight', pad_inches=0)

    # plt.figure()
    # for d in data:
    #     plt.plot(np.cumsum(d["fitness"]))

    plt.show()


if __name__ == '__main__':
    import argh
    argh.dispatch_commands([get_trajectory,
                            ])
