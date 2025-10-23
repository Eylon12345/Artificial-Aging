import numpy as np
import os
import matplotlib.pyplot as plt
from ipywidgets import interact
import pandas as pd
from functools import partial

from mindcraft import World
from aging import rollout_lifetime, load_data, save_data

max_steps = 1000            # maximum number of environment steps (action perception cycles)
num_episodes = 10           # number of independent "lifes" (successively executed to gather data)
noise_level = 0.15
competency_level = 1.0

Nx, Ny = 16, 16
N_embd = 8
N_state = 4
N_act = 4

logging = dict(log_fields=('reward', 'state', 'embedding', 'action', 'intervened'),
               log_foos={'state': f'observation[:, 0].reshape({Nx}, {Ny}, -1)',
                         'embedding': f'self.agent.embedding.detach().numpy().reshape({Nx}, {Ny}, -1) if self.agent.embedding is not None else np.zeros(({Nx}, {Ny}, {N_embd}))',
                         'action': f'action.reshape({Nx}, {Ny}, -1) if not len(action) == {N_act} else np.zeros(({Nx}, {Ny}, {N_act}))',                 
                         'intervened': '{k: v[-1] for k, v in self.env.intervened.items() if len(v)} if hasattr(self.env, "intervened") else False',
                         }
              )


def get_movie(rollou_data, export_path=None, max_steps=1000):
    import matplotlib.pyplot as plt
    from matplotlib import animation
    from ipywidgets import Button
    from IPython.display import HTML, display
    
    # Sample time series data (replace with your data)
    times = range(max_steps)
    
    # Configure plot
    fig, axes = plt.subplots(1, num_episodes, sharex=True, sharey=True, figsize=(15, 3))
    fig = plt.figure(figsize=(15,3))
    subfigs = fig.subfigures(nrows=1, ncols=num_episodes)

    ims = []
    axes = []
    for i in range(num_episodes): 
        ax = subfigs[i].subplots()
        data = rollou_data["state"][i][0, :, :, :3].argmax(axis=-1)
        ims.append(ax.imshow(data, cmap='Spectral_r', vmin=0, vmax=2, interpolation="none"))
        ax.set_xticks([])
        ax.set_yticks([])

        axes.append(ax)

    axes[0].set_ylabel(f"t = {times[i]}")

    
    def update(i):
        t_eff = i * max_steps // len(times)
        try:
            interventions = [
                [{"left_eye": left_step, "right_eye": right_step, "mouth": mouth_step} for left_step, right_step, mouth_step in zip(*organ_interventions)]
                for organ_interventions in zip(rollou_data["left_eye_interventions"], rollou_data["right_eye_interventions"], rollou_data["mouth_interventions"])
            ]
        except KeyError:
            interventions = []
        
        for sf, im, ax, data, intervention in zip(subfigs, ims, axes, rollou_data["state"], interventions):  # episodes
            im.set_array(data[t_eff, ..., :3].argmax(axis=-1))
            ax.set_title(f"Time: {t_eff}")

            intervened = False
            for k, c in zip(("left_eye", "right_eye", "mouth"),
                            ("blue", "green", "red")):
                if interventions and isinstance(intervention[t_eff], dict) and intervention[t_eff][k]:
                    sf.set_facecolor(c)
                    intervened = True
                    
            if not intervened:
                sf.set_facecolor("white")
                
    
    ani = animation.FuncAnimation(fig, update, frames=len(times), interval=100)

    if export_path:
        writervideo = animation.FFMpegWriter(fps=60) 
        print(f"export to {export_path}")
        ani.save(export_path, writer=writervideo) 
    
    # Add pause button with output_toggle
    play_button = Button(description="Play/Pause", layout={'width': '100px'})
    
    def toggle_play(b):
        if ani.event_source.running:
            ani.event_source.stop()
            b.description = "Play"
        else:
            ani.event_source.start()
            b.description = "Pause"

    play_button.on_click(toggle_play)
    display(play_button)
    display(HTML(ani.to_jshtml()))


def plot_reward(recovery_data, data_default=None, fig=None, ax=None):
    if ax is None:
        fig = plt.figure(figsize=(12, 4))
        ax = plt.gca()
        
    # plt.plot(data_trained["cumulative_reward"].T, color="lightgray")
    ax.plot(recovery_data["cumulative_reward"].mean(axis=0), label="Mean", linewidth=2, zorder=2)    
    ax.plot(recovery_data["cumulative_reward"][:-1, :].T, color="lightgreen", zorder=1)  # label="single recovered")
    ax.plot(recovery_data["cumulative_reward"][-1, :], color="lightgreen", label="Individuals", zorder=1)
    if data_default is not None:
        ax.plot(data_default["cumulative_reward"].mean(axis=0), label="Baseline", color="black", linestyle="--")
    ax.set_ylabel("Fitness")
    ax.set_xlabel("Time Steps")
    ax.set_xlim([0, 1000])
    ax.legend()
    ax.grid()

    return fig


def get_organ_fitness(state_trajectory, left_eye, right_eye, mouth, left_eye_socket=None, right_eye_socket=None, mouth_socket=None):
    fitness = {}
    
    for key, mask, socket, val, embd in (("left_eye", left_eye, left_eye_socket, 2, 1),
                                         ("right_eye", right_eye, right_eye_socket, 2, 1),
                                         ("mouth", mouth, mouth_socket, 2, 1)):
        organ_state = state_trajectory[:, mask[:, 1], mask[:, 0]]
        organ_fitness = (organ_state == val).reshape(len(state_trajectory), len(mask))

        socket_fitness = 0
        if socket is not None:
            socket_state = state_trajectory[:, socket[:, 1], socket[:, 0]]
            socket_fitness = (socket_state != embd).reshape(len(state_trajectory), len(socket))
            socket_fitness = np.sum(socket_fitness, axis=-1) / len(mask)

        fitness[key] = organ_fitness.mean(axis=-1) - socket_fitness

    return fitness


def get_intervention_times(data):
    # intervention = data["intervened"]
    rc = []
    for episode in range(num_episodes):
        rc.append({})
        if not all (f"{k}_interventions" in data for k in ("left_eye", "right_eye", "mouth")):
            continue

        for k in ("left_eye", "right_eye", "mouth"):
            ie = data[f"{k}_interventions"][episode]
            k_interventions = np.where(ie)[0] # [t for t, i in enumerate(ie) if i]
            rc[-1][k] = k_interventions

    return rc


def plot_organ_fitness(data, **organs):
    fitness = []
    for i, episode_state_trajectory in enumerate(data["state"]):
        fitness.append(get_organ_fitness(episode_state_trajectory[..., :3].argmax(axis=-1), **organs))
    
    fitness = pd.DataFrame(fitness)
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True, sharey=True)
    [axes[0].plot(l, color="cornflowerblue", zorder=0) for l in fitness["left_eye"]]
    axes[0].plot(fitness["left_eye"].values[-1], color="cornflowerblue", zorder=0, label="Individual Score")
    axes[0].plot(fitness["left_eye"].mean(), color="black", zorder=1, label="Mean Score")
    axes[0].set_ylabel("Left Eye - Score")
    
    [axes[1].plot(l, color="mediumseagreen", zorder=0) for l in fitness["right_eye"][:-1]]
    axes[1].plot(fitness["right_eye"].values[-1], color="mediumseagreen", zorder=0, label="Individual Score")
    axes[1].plot(fitness["right_eye"].mean(), color="black", zorder=1, label="Mean Score")
    axes[1].set_title("")
    axes[1].set_ylabel("Right Eye - Score")
    
    [axes[2].plot(l, color="salmon", zorder=0) for l in fitness["mouth"][:-1]]
    axes[2].plot(fitness["mouth"].values[-1], color="salmon", zorder=0, label="Individual Score")
    axes[2].plot(fitness["mouth"].mean(), color="black", zorder=1, label="Mean Score")
    axes[2].set_xlabel("Time Steps")
    axes[2].set_ylabel("Mouth - Score")

    [ax.set_xlim([0, 1000]) for ax in axes]

    intervention_times = get_intervention_times(data)
    interventions_per_episode = {"left_eye": [], "right_eye": [], "mouth": []}
    for episode in range(num_episodes):
        intervention_count = {k: 0 for k in interventions_per_episode.keys()}
        for ax, k, c in zip(axes,
                            interventions_per_episode.keys(),
                            ("blue", "green", "red")):
            
            for intervention_time in intervention_times[episode].get(k, []):
                ax.axvspan(intervention_time, intervention_time + 1, alpha=0.3, color=c, linewidth=0.)
                intervention_count[k] += 1

        for k, v in intervention_count.items():
            interventions_per_episode[k].append(v)

    [ax.legend() for ax in axes]

    total_count = []
    for i in range(num_episodes):
        total_count.append(sum([v[i] for v in interventions_per_episode.values()]))

    interventions_per_episode["total"] = total_count

    fig2 = plt.figure(figsize=(12, 2))
    multiplier = 0
    width = 0.1
    offset = -0.125 - 0.0625
    for k, c in zip(interventions_per_episode.keys(),
                    ("cornflowerblue", "mediumseagreen", "salmon", "gray")):
        plt.bar(offset + np.arange(len(interventions_per_episode[k])), interventions_per_episode[k], label=k, color=c, width=width)
        offset += 0.125

    plt.xlabel("Episodes")
    plt.ylabel("Intervention Counts")
    ax1 = plt.gca()
    ax1.set_xticks(range(10), [i+1 for i in range(10)])
    ax2 = ax1.twinx()
    ax2.plot(data["cumulative_reward"][:, -1], marker="o", color="magenta", linewidth=0.5)
    ax2.set_ylabel('Final Fitness', color='magenta')
    ax2.set_ylim([0, 260])

    ax1.legend()
    ax1.grid()

    return fig, fig2


def set_initial_state(world, initial_state, organ_mask, socket_mask=None,
                      injection_mask=None, target_state=2, socket_target=1, threshold=0., development=150, horizon=150, reset_memory=True,
                      label="left_eye", only_correct_wrongs=False,
                     ):
    try:
        intervened = world.env.intervened
    except:
        intervened = {"left_eye": [], "right_eye": [], "mouth": []}

    state = world.env.state.get_val(channel="type")
    organ_state = state[organ_mask[:, 1], organ_mask[:, 0]].argmax(axis=-1)

    socket_loss = 0
    if socket_mask is not None:
        socket_state = state[socket_mask[:, 1], socket_mask[:, 0]].argmax(axis=-1)
        socket_loss = (socket_state != socket_target).sum() / len(organ_mask)
    
    intervene = world.env.simulation_step >= development
    intervene &= not any(intervened[label][-horizon:])
    intervene &= (np.mean(organ_state == target_state) - socket_loss) <= threshold
    
    if intervene:
        if injection_mask is None:
            if socket_mask is not None:
                injection_mask = np.concatenate([organ_mask, socket_mask])
            else:
                injection_mask = organ_mask

        if only_correct_wrongs:
            wrongs = state[injection_mask[:, 1], injection_mask[:, 0]].argmax(axis=-1) != target_state
            mask = injection_mask[wrongs]

            wrongs = state[injection_mask[:, 1], injection_mask[:, 0]].argmax(axis=-1) != target_state

        else:
            mask = injection_mask
            
        init_values = initial_state[mask[:, 1], mask[:, 0], :]
        if init_values.size > 0:
            world.env.state.array[mask[:, 1], mask[:, 0], :] = init_values

        # initial_type_val = initial_state[mask[:, 1], mask[:, 0], :3]
        # world.env.state.set_state(mask[:, ::-1], value=initial_type_val, channel="type")
        # initial_hidden_val = initial_state[mask[:, 1], mask[:, 0], 3:]
        # world.env.state.set_state(mask[:, ::-1], value=initial_hidden_val, channel="hidden")

        if reset_memory and world.agent.policy_module._states is not None:
            hidden_states = world.agent.policy_module._states[0].clone().reshape(Nx, Ny, N_state)
            hidden_states[mask[:, 1], mask[:, 0]] = 0.
            world.agent.policy_module._states[0][...] = hidden_states.reshape(-1, N_state)

    intervened[label].append(intervene)
    world.env.intervened = intervened


def run_recovery(threshold, development, horizon, reset_memory, inject_initial, world_path, left_eye, right_eye, mouth, left_eye_socket=None, right_eye_socket=None, mouth_socket=None, only_correct_wrongs=False):
    recover_kwargs = dict(threshold=threshold, development=development, horizon=horizon, reset_memory=reset_memory, only_correct_wrongs=only_correct_wrongs)
    recover_left_eye = partial(set_initial_state, organ_mask=left_eye, socket_mask=left_eye_socket, label="left_eye", **recover_kwargs)
    recover_right_eye = partial(set_initial_state, organ_mask=right_eye, socket_mask=right_eye_socket, label="right_eye", **recover_kwargs)
    recover_mouth = partial(set_initial_state, organ_mask=mouth, label="mouth", **recover_kwargs)
    
    if inject_initial is not None:
        injection_state = inject_initial.numpy()
    
    else:
        mature_state = np.zeros((Nx, Ny, N_state))
        mature_state[left_eye[:, 1],  left_eye[:, 0], :3] = [-3., -3., 3.]
        mature_state[right_eye[:, 1], right_eye[:, 0], :3] = [-3., -3., 3.]
        mature_state[mouth[:, 1],     mouth[:, 0], :3] = [-3., -3., 3.]
        injection_state = mature_state
    
    schedule = {recover_left_eye: [injection_state for _ in range(max_steps)],
                recover_right_eye: [injection_state for _ in range(max_steps)],
                recover_mouth: [injection_state for _ in range(max_steps)],
               }
    
    world_kwargs = dict(max_steps=max_steps,
                        n_episodes=num_episodes,
                        schedule=schedule,
                        verbose=True,
                        render=False,
                        **logging
                        )
    
    world = World.make(world_path, **world_kwargs)
    world.agent.sampling = competency_level
    world.env.state.noise_level = noise_level
    world.env.task.stagnation_cost = 0.
    world.env.task.completion_reward = 0.

    world.env.simulation_step = 0
    for _ in range(num_episodes):
        world.rollout()
        world.env.simulation_step = 0
        del world.env.intervened
    
    recovery_data = world.log_history
    recovery_data = {k: np.stack([d[k] for d in recovery_data], axis=0) for k in logging["log_fields"]}
    
    if "reward" in logging["log_fields"] and world.env.incremental_reward:
        recovery_data["cumulative_reward"] = np.cumsum(recovery_data["reward"], axis=1)
    
    if 'state' in logging["log_fields"]:
        recovery_data['type'] = np.argmax(recovery_data['state'][..., :3], axis=-1)

    left_eye_interventions = np.array([[timestep["left_eye"] for timestep in episode] for episode in recovery_data["intervened"]])
    right_eye_interventions = np.array([[timestep["right_eye"] for timestep in episode] for episode in recovery_data["intervened"]])
    mouth_interventions = np.array([[timestep["mouth"] for timestep in episode] for episode in recovery_data["intervened"]])

    recovery_data["left_eye_interventions"] = left_eye_interventions
    recovery_data["right_eye_interventions"] = right_eye_interventions
    recovery_data["mouth_interventions"] = mouth_interventions
    del recovery_data["intervened"]
    
    return recovery_data


def get_filename(recover, threshold, development, horizon, reset_memory, inject_initial, only_correct_wrongs, postfix="h5", **kwargs):
    inject = "init" if inject_initial is not None else "target"
    correct = "wrongs" if only_correct_wrongs else "all"
    rnnstate = "reset" if reset_memory else "keep"
    return f"recover_{recover}-threshold_{threshold}-development_{development}-horizon_{horizon}-rnnstate_{rnnstate}-inject_{inject}-correct_{correct}.{postfix}"


def get_movie_name(recover, aging_path, **kwargs):
    fname = get_filename(recover=recover, postfix="mp4", **kwargs)
    return os.path.join(aging_path, fname)


def plot_fitness(data, aging_path, left_eye, right_eye, mouth, dpi=300, recover="socket", data_default=None, **kwargs):
    fig_reward = plot_reward(data, data_default=data_default)
    fig_organ_fitness, fig_count = plot_organ_fitness(data, left_eye=left_eye, right_eye=right_eye, mouth=mouth)

    fname_base = "aging-recovery-eye-and-socket-loss-" + "-".join([f"{k}_{v}" for k, v in kwargs.items()])
    fname_base = get_filename(recover=recover, postfix="png", **kwargs)
    
    fname_reward = os.path.join(aging_path, fname_base.replace(".png", "-reward.png"))
    fname_organ_fitness = os.path.join(aging_path, fname_base.replace(".png", "-organ-fitness.png"))
    fname_intervention_count = os.path.join(aging_path, fname_base.replace(".png", "-intervention-count.png"))

    print(f"export figure {fname_reward}")
    fig_reward.savefig(fname_reward, dpi=dpi)
    print(f"export figure {fname_organ_fitness}")
    fig_organ_fitness.savefig(fname_organ_fitness, dpi=dpi)
    print(f"export figure {fname_intervention_count}")
    fig_count.savefig(fname_intervention_count, dpi=dpi)