import os
from musepy import experiment as mp
from typing import Union

WORLD_LOG_FIELDS = ["reward", "state", "info"]
WORLD_LOG_FOOS = {"state": "self.env.state_array.copy()",
                  "info": "self.env.info",
                  }

#EXPERIMENT_FOLDER = os.path.abspath(os.path.dirname(__file__))
#PROJECT_NAME = "multiscale-evolution"
#PROJECT_FOLDER = os.path.join(PROJECT_NAME.join(EXPERIMENT_FOLDER.split(PROJECT_NAME)[:-1]), PROJECT_NAME)
#EXPERIMENT_NAME = EXPERIMENT_FOLDER.split("/")[-1]
EXPERIMENT_DATA_PATH = "data"

AGENTS_FOLDER = "agents"
TASKS_FOLDER = "tasks"
DEFAULT_CONFIG = "configs/train.yml"

def get_files(task_id, agent, destination=EXPERIMENT_DATA_PATH, agents=AGENTS_FOLDER, prefix=None):
    """ Get the agent file and destination path. """
    flags_dir = os.path.dirname(os.path.realpath(__file__))
    agents_dir = os.path.join(flags_dir, agents)
    agent_file = os.path.join(agents_dir, agent)

    args = [task_id] if not prefix else [task_id, prefix]
    dst_path = mp.get_dst_path(*args, path=destination, agent=agent)

    return args, agent_file, dst_path

def get_checkpoint_file(continue_training: Union[bool, dict, str]):
    """ Convert the `continue_training` argument to a boolean and a dictionary,
        specifying whether to continue training and which checkpoint to load.

    :param continue_training: A boolean, a dictionary, or a string, specifying the checkpoint to load.
                              In case of a boolean, `True` loads the latest checkpoint, `False` loads a new model.
                              In case of a dictionary, the keys are "runs" and "gens" specifying the run and generation
                              to load (use -1 for latest runs or gens, or use {} for the latest checkpoint).
                              In case of a string, it is parsed as a json string.
    :return: A tuple of (continue_training, checkpoint),
             where `continue_training` is a boolean,
             and `checkpoint` is a dictionary.
    """
    if continue_training:
        if isinstance(continue_training, bool):
            return continue_training, None

        if isinstance(continue_training, dict):
            return True, continue_training

        if isinstance(continue_training, str):
            import json
            return True, json.loads(continue_training)

    return False, None
