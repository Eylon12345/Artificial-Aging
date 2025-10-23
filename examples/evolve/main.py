import os
from musepy import experiment as mp
from helper import EXPERIMENT_DATA_PATH, AGENTS_FOLDER, TASKS_FOLDER, DEFAULT_CONFIG, WORLD_LOG_FIELDS, WORLD_LOG_FOOS
from helper import get_files, get_checkpoint_file


def train(task_id,
          agent,
          config=DEFAULT_CONFIG,
          destination=EXPERIMENT_DATA_PATH,
          continue_training=None,
          agents=AGENTS_FOLDER,
          tasks=TASKS_FOLDER,
          prefix="",
          ):
    """ Train an agent on a task.

    :param task_id: chose e.g., "czech_8x8.json", "french_6x6.json", ... from the `tasks` folder.
    :param agent: chose yml-file from `agents` folder.
    :param config: chose yml-file from `configs` folder.
    :param destination: [Optional] path to store the results.
    :param continue_training: [Optional] continue training from a checkpoint (use `True` for latest or specify as tuple `(run, generation)`).
    :param agents: [Optional] folder containing the agents.
    :param tasks: [Optional] folder containing the tasks.
    :param prefix: [Optional] prefix for the file-naming in the destination folder.
    """

    args, agent_file, dst_path = get_files(task_id, agent, destination, agents, prefix)
    assert os.path.exists(agent_file), f"Agent file {agent_file} does not exist."

    continue_training, checkpoint = get_checkpoint_file(continue_training)
    return mp.train(task_id=task_id,
                    agent=agent_file,
                    config=config,
                    tasks=tasks,
                    path=dst_path,
                    new_model=not continue_training,
                    checkpoint=checkpoint,
                    )


def test(task_id,
         agent,
         mode="test",
         task_num=0,
         render_config=None,
         destination=EXPERIMENT_DATA_PATH,
         agents=AGENTS_FOLDER,
         tasks=TASKS_FOLDER,
         checkpoint="{}",
         prefix="",
         quiet=False,
         delay=0.05,
         max_steps=50,
         n_episodes=5,
         disable_render=False,
         log_fields=WORLD_LOG_FIELDS,
         log_foos=WORLD_LOG_FOOS,
         ):
    """ Test an agent on a task.

    :param task_id: chose e.g., "czech_8x8.json", "french_6x6.json", ... from the `tasks` folder.
    :param agent: chose yml-file from `agents` folder.
    :param mode: [Optional] chose from "test" or "train".
    :param task_num: [Optional] chose the task number (TODO).
    :param render_config: [Optional] chose yml-file from `configs` folder (None is fine too).
    :param destination: [Optional] path to store the results.
    :param agents: [Optional] folder containing the agents.
    :param tasks: [Optional] folder containing the tasks.
    :param checkpoint: [Optional] continue training from a checkpoint (use `True` for latest or specify as dictionary `{runs: int, gens: int}`; -1 will load the latest).
    :param prefix: [Optional] prefix for the file-naming in the destination folder.
    """

    print(f"Testing Task '{task_id}' with Agent '{agent}' in '{mode}'-mode" +
          f" with prefix `{prefix}`" * (prefix is not None) +
          ".")


    args, agent_file, dst_path = get_files(task_id, agent, destination, agents, prefix)
    assert os.path.exists(agent_file), f"Agent file {agent_file} does not exist."
    return mp.test(task_id=task_id,
                   mode=mode,
                   task_num=task_num,
                   path=dst_path,
                   tasks=tasks,
                   render_config=render_config,
                   checkpoint=checkpoint,
                   quiet=quiet,
                   delay=delay,
                   max_steps=max_steps,
                   n_episodes=n_episodes,
                   disable_render=disable_render,
                   log_fields=log_fields,
                   log_foos=log_foos,
                   )


def progress(task_id,
             agent,
             destination=EXPERIMENT_DATA_PATH,
             agents=AGENTS_FOLDER,
             prefix="",
            ):
    """ Monitor the training progress of the agent.

    :param task_id: chose e.g., "czech_8x8.json", "french_6x6.json", ...
    :param agent: chose yml-file from `agents` folder.
    :param destination: [Optional] path to store the results.
    :param agents: [Optional] folder containing the agents.
    :param prefix: [Optional] prefix for the file-naming in the destination folder.
    :param evaluate_structural_fitness: [Optional] evaluate the genotypic fitness of the agent.
    """

    args, agent_file, dst_path = get_files(task_id, agent, destination, agents, prefix)
    assert os.path.exists(agent_file), f"Agent file {agent_file} does not exist."

    log_file = os.path.join(dst_path, "agent.log")
    return mp.progress(log_file)


def checkpoints(task_id,
                agent,
                mode="test",
                task_num=0,
                destination=EXPERIMENT_DATA_PATH,
                agents=AGENTS_FOLDER,
                tasks=TASKS_FOLDER,
                prefix="",
                verbose=False,
                show_progress=False,
                show_states="palette",
                ):
    """ Evaluate an agent by loading a checkpoint.

    :param task_id: chose e.g., "czech_8x8.json", "french_6x6.json", ... from the `tasks` folder.
    :param agent: chose yml-file from `agents` folder.
    :param mode: [Optional] chose from "test" or "train".
    :param task_num: [Optional] chose the task number (TODO).
    :param destination: [Optional] path to store the results.
    :param agents: [Optional] folder containing the agents.
    :param tasks: [Optional] folder containing the tasks.
    :param prefix: [Optional] prefix for the file-naming in the destination folder.
    :param verbose: [Optional] print additional information.
    :param show_progress: [Optional] show the progress of the agent.
    :param show_states: [Optional] show the states of the agent (choose from "palette" or "rgb").
    """

    print(f"Checkpointing Task '{task_id}' with Agent '{agent}' in '{mode}'-mode" +
          f" with prefix `{prefix}`" * (prefix is not None) +
          ".")


    args, agent_file, dst_path = get_files(task_id, agent, destination, agents, prefix)
    assert os.path.exists(agent_file), f"Agent file {agent_file} does not exist."
    return mp.checkpoints(task_id=task_id,
                          path=dst_path,
                          mode=mode,
                          task_num=task_num,
                          tasks=tasks,
                          verbose=verbose,
                          show_progress=show_progress,
                          show_states=show_states,
                          )


if __name__ == "__main__":
    import argh
    argh.dispatch_commands([
        train,
        test,
        progress,
        checkpoints,
    ])