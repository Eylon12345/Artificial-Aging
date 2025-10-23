import numpy as np
import itertools as it
from torch import nn, tensor
from scipy.spatial import distance_matrix
from mindcraft.io import Repr
from mindcraft.util.tensor import to_one_hot, to_categorical
from musepy.util.state import State

class Task(Repr):
    """ A default Task for the Multicellular Automaton. """
    REPR_FIELDS = ()

    def __init__(self, **repr_kwargs):
        """ Constructs a Task """
        Repr.__init__(self, **repr_kwargs)

    def reset(self):
        pass

    @classmethod
    def default(cls):
        return cls()

    def init_task(self, **kwargs):
        self.reset()
        return {}

    def evaluate(self, env, **kwargs):
        """ Evaluate the task on the given (cells, states, grid) configuration.

        :param env: A HybridNCA instance.
        """
        reward = 0.
        done = self.is_done(reward, env.n_cells)
        info = {}
        return reward, done, info

    def is_done(self, reward, num_cells):
        return num_cells == 0


class StateTask(Task):
    """ A simple environment with a corpus of spatial state-patterns with cross-entropy evaluation
    (c.f. the ARC challenge). """
    REPR_FIELDS = ("corpus",
                   "palette",
                   "mode",
                   "alive_threshold",
                   "one_hot",
                   "eps",
                   "target_cost",
                   "individual_cost",
                   "individual_reward",
                   "stagnation_cost",
                   "completion_reward",
                   "num_diff_cost",
                   "init_target",
                   "schedule",
                   *Task.REPR_FIELDS
                   )

    RANDOM_SCHEDULE = "random"
    ITER_SCHEDULE = "iter"
    TASKS_SCHEDULE = "tasks"
    NO_SCHEDULE = "no"
    SCHEDULE = (RANDOM_SCHEDULE, ITER_SCHEDULE, NO_SCHEDULE, TASKS_SCHEDULE, None, "None", "null", False)

    HARDMAX_COST = "hardmax"
    SOFTMAX_COST = "softmax"
    MSE_COST = "mse"
    TARGET_COST = (HARDMAX_COST, SOFTMAX_COST, MSE_COST)

    def __init__(self, corpus, palette, mode=None,
                 one_hot=True, target_cost=HARDMAX_COST, eps=2**-100,
                 stagnation_cost=0., individual_cost=0.1, completion_reward=1., num_diff_cost=0.,
                 init_target=False, schedule=None, alive_threshold=None,
                 individual_reward=None,
                 **kwargs):
        """ Constructs a StateTask instance

        :param corpus: A dictionary of the form {mode: [{'input': [...], 'output': [...], ...}, ...}, specifying for
                       a giving task mode, e.g., 'train' or 'test' (up to the user), a list of dictionaries
                       that define the 'input' (`start_state`) and 'output' (`target_state`) pairs that are compared
                       in the evaluation function. Note that {key: value} pairs may be specified in each task.
                       The `input` and `output` arrays must be an `ndim` array `[[..., state_ij, ...]]` specifying
                       an integer `state` in `[0, len(palette)-1]` the `ij`-th element. Both `input` and `output` are
                       transformed via `one_hot` encoding and can be retrieved by the `start_state` and `target_state`
                       properties.
        :param palette: Either a list of colors for all possible channels of the `input` and `output`, or the maximum
                        number of state channels (the palette will be transformed into a color-palette then).
        :param mode: The `corpus[mode]` that is initialized and evaluated.
        :param schedule: Optional choice of `StateTask.SCHEDULE` (e.g., `None`, "random", "tasks", ...) to define
                         a task scheduling in the `init_task` call, defaults to `None`.
                         `StateTask.RANDOM_SCHEDULE` will pick a random task from `corpus[mode]`.
                         `StateTask.TASKS_SCHEDULE` will iteratively pick the next task from `corpus[mode]`, starting
                         from `self.task_num` (that defaults to 0).
        :param one_hot: Boolean to enable/disable one-hot comparison between target pattern and MuCA channel states
                        in the evaluation method.
        :param eps: Numerical offset for evaluating `cross_entropy` loss.
        :param num_diff_cost: (WIP) The evaluation cost for having the wrong number of cells per channel compared to the
                              `target_state`, defaults to 0.
        :param individual_reward: The evaluation reward for a cell being at the correct location and of correct type.
        :param individual_cost: The relative cost for the cross-entropy loss in the evaluate method, defaults to 0.1.
        :param stagnation_cost: The relative cost if cell types don't change if the pattern is not complete, defaults to 0.
        :param completion_reward: The relative reward for finding the correct final state, defaults to 10.
        :param init_target: Boolean to enable/disable NCA initialization from the `target_state` (output) instead of
                            the `start_state` (input). Helpful for visualization, defaults to False.
        :param kwargs: Keyword arguments forwarded to `Task` superclass constructor.
        """
        self._corpus = None
        self.corpus = corpus
        self.palette = palette
        self.mode = mode or 0
        self.task_num = None
        self.one_hot = one_hot
        self.target_cost = target_cost
        self.eps = eps
        self.individual_cost = individual_cost
        self.individual_reward = individual_reward if individual_reward is not None else individual_cost
        self.stagnation_cost = stagnation_cost
        self.completion_reward = completion_reward
        self.num_diff_cost = num_diff_cost
        self.init_target = init_target
        self.schedule = schedule
        self.alive_threshold = alive_threshold

        to_list = kwargs.get("to_list", [])
        to_list.extend(["channels", ])
        kwargs["to_list"] = tuple(to_list)

        self._start_mask = None
        self._start_state = None
        self._start_coords = None
        self._num_start = None

        self._target_mask = None
        self._target_state = None
        self._target_coords = None
        self._num_target = None

        self._prev_reward = None

        self._next_task = int(np.min(self.task_num)) if self.task_num is not None else 0
        Task.__init__(self, **kwargs)

    @property
    def corpus(self) -> dict:
        return self._corpus

    @corpus.setter
    def corpus(self, value: dict):
        self._corpus = {
            mode_key: [
                {'input': np.asarray(task_i['input']), 'output': np.asarray(task_i['output']),
                 **{k: v for k, v in task_i.items() if k not in ('input', 'output')}}
                for task_i in tasks
            ]
            for mode_key, tasks in value.items()
        }

    def reset(self):
        if isinstance(self.palette, int):
            from matplotlib.pyplot import get_cmap
            self.palette = get_cmap('terrain', self.palette)

        start_state = np.copy(self.corpus[self.mode][self.task_num]['input'])
        target_state = np.copy(self.corpus[self.mode][self.task_num]['output'])
        if self.alive_threshold is not None:
            self._start_mask = (start_state >= (self.alive_threshold or 0))
            start_state[~self._start_mask] = self.alive_threshold

            self._target_mask = (target_state >= (self.alive_threshold or 0))
            target_state[~self._target_mask] = self.alive_threshold
        else:
            self._start_mask = np.ones_like(start_state, dtype=bool)
            self._target_mask = np.ones_like(target_state, dtype=bool)

        kws = dict(num_classes=len(self.palette), reshape=False)
        self._start_state = to_one_hot(start_state, **kws)
        self._start_state[~self._start_mask] = self.alive_threshold
        self._num_start = np.sum(self._start_mask)

        self._target_state = to_one_hot(target_state, **kws)
        self._target_state[~self._target_mask] = self.alive_threshold
        self._num_target = np.sum(self._target_mask)

        self._prev_reward = None

    def to_dict(self):
        dict_repr = Task.to_dict(self)
        dict_repr['corpus'] = {
            mode_key: [
                {k: np.asarray(v).tolist() if isinstance(v, (np.ndarray, list)) else k for k, v in task_i.items()}
                for task_i in tasks
            ]
            for mode_key, tasks in dict_repr.get('corpus', {}).items()
        }
        return dict_repr

    @property
    def target_state(self):
        """ A one-hot target array state of the corpus """
        return self._target_state

    @property
    def start_state(self):
        return self._start_state

    @property
    def num_tasks(self):
        return len(self.corpus[self.mode])

    def init_task(self, mode=None, task_num=None, bounds=None):
        # optional mode update
        if mode is not None:
            self.mode = mode

        # task scheduling
        if task_num is not None:
            self.task_num = task_num
        elif self.schedule == self.RANDOM_SCHEDULE:
            self.task_num = np.random.randint(0, len(self.corpus[self.mode]))
        elif self.schedule == self.TASKS_SCHEDULE:
            self._next_task = int(self._next_task % self.num_tasks)
            self.task_num = self._next_task
            self._next_task += 1

        # init start_state and target_state
        self.reset()

        # evaluate NCA init attributes
        state_channel_initialize = self.get_state_channel_initialize()
        grid_size = self.get_grid_size()
        if bounds is not None:
            if self.one_hot:
                state_channel_initialize = state_channel_initialize * (bounds[1] - bounds[0]) + bounds[0]
            else:
                state_channel_initialize = state_channel_initialize / (self.num_classes - 1) * (bounds[1] - bounds[0]) + bounds[0]
        return {"state": {"initialize": {State.TYPE_CHL: state_channel_initialize}},
                "grid": {"size": grid_size},
                "render": {"palette": self.palette},
                }

    @property
    def ndim(self):
        return len(self.target_state.shape) - 1

    def get_grid_size(self):
        if self.init_target:
            return self.target_state.shape[:self.ndim]
        return self.start_state.shape[:self.ndim]

    def get_state_channel_initialize(self):
        if self.init_target:
            if self.one_hot:
                state = self.target_state
            else:
                state = np.copy(self.corpus[self.mode][self.task_num]['output'])
                state[~self._target_mask] = self.alive_threshold
        else:
            if self.one_hot:
                state = self.start_state
            else:
                state = np.copy(self.corpus[self.mode][self.task_num]['input'])
                state[~self._start_mask] = self.alive_threshold
        return state

    @property
    def num_channels(self):
        if self.one_hot:
            return self.target_state.shape[-1]
        return 1

    @property
    def num_classes(self):
        return len(self.palette)

    @staticmethod
    def hardmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_argmax = np.argmax(x, axis=axis)
        y = np.zeros_like(x)
        np.put_along_axis(y, np.expand_dims(x_argmax, axis=axis), 1, axis=axis)
        return y

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        nom = np.exp(x - np.max(x))
        return nom / np.expand_dims(np.sum(nom, axis=axis), axis)

    def get_state(self, env):
        state = env.state.get_val(channel=State.TYPE_CHL)

        if not self.one_hot:
            bounds = env.state.bounds
            if bounds is not None:
                state = state / (bounds[1] - bounds[0]) + bounds[0]
                state *= (self.num_classes - 1)
            state = np.rint(np.clip(state, 0, self.num_classes - 1)).astype(int)
            state = to_one_hot(state, num_classes=self.num_classes, reshape=False)
            return state.reshape(self.target_state.shape)

        if self.target_cost == self.HARDMAX_COST:
            return self.hardmax(state)
        elif self.target_cost == self.SOFTMAX_COST:
            return self.softmax(state)
        elif self.target_cost == self.MSE_COST:
            bounds = env.state.bounds
            if bounds is not None:
                state = state / (bounds[1] - bounds[0]) + bounds[0]
            state = np.rint(np.clip(state, 0, 1)).astype(float)
            return state

        raise NotImplementedError(f"target_cost: {str(self.target_cost)}.")

    def cross_entropy(self, y_true, y_pred, axis=None):  # CE
        #if self.one_hot:
        # return -np.sum(y_true * np.log(y_pred[..., :self.num_channels] + self.eps), axis=axis)
        #return -np.sum(y_true * np.log(y_pred[..., :self.num_channels]), axis=axis)

        if self.target_cost == self.HARDMAX_COST:
            # cross_entropy_cost = ~np.all(y_true == y_pred, axis=-1) + 0.551444713932051
            cross_entropy_cost = 2.*(~np.all(y_true == y_pred, axis=-1) - 0.5)

        elif self.target_cost == self.SOFTMAX_COST:
            y_pred, y_true = tensor(y_pred).transpose(-1, -2), tensor(y_true).transpose(-1, -2)
            cross_entropy_cost = nn.CrossEntropyLoss(reduction='none')(y_pred, y_true).numpy()

            if self.alive_threshold is not None:
                cross_entropy_cost[~self._target_mask] = 1.551444713932051

            cross_entropy_cost = 2.*(cross_entropy_cost - 1.051444713932051)

        elif self.target_cost == self.MSE_COST:
            cross_entropy_cost = (np.square(y_true - y_pred)).mean(axis=axis)

        else:
            raise NotImplementedError(f"target_cost: {str(self.target_cost)}.")

        return cross_entropy_cost

    def cross_entropy_grad(self, y_true, y_pred):  # CE Jacobian
        #if self.one_hot:
        return -y_true / (y_pred[..., :self.num_channels] + self.eps)
        #return -y_true / y_pred[..., :self.num_channels]

    def evaluate(self, env, **kwargs):
        """ Evaluate the task on the given (cells, states, grid) configuration.

        :param env:  A Hybrid NCA instance (HybridNCA).
        """
        cross_entropy_cost = 0.
        state = self.get_state(env)
        try:
            cross_entropy_cost = self.cross_entropy(self.target_state, state)

        except ValueError as ve:
            import warnings
            warnings.warn(f"SHAPE MISMATCH LIKELY - NOT IMPLEMENTED:\n{str(ve)}")

        finally:
            completion_reward = 0.0
            if np.array_equal(self.target_state, state):
                completion_reward = 1.

        reward, done, info = Task.evaluate(self, env=env, **kwargs)

        info["individual_reward"] = 0.
        info["completion_reward"] = 0.
        info["stagnation_cost"] = 0.
        info["num_diff_cost"] = 0.

        individual_reward = np.zeros_like(cross_entropy_cost)
        if isinstance(cross_entropy_cost, np.ndarray):
            if self.individual_reward != self.individual_cost:
                ce = cross_entropy_cost
                individual_reward -= (ce * (ce < 0)) * self.individual_reward
                individual_reward -= (ce * (ce > 0)) * self.individual_cost
            else:
                individual_reward -= cross_entropy_cost * self.individual_cost

        info["individual_reward"] = individual_reward
        reward += np.sum(individual_reward)

        if completion_reward:
            info["completion_reward"] = completion_reward * self.completion_reward
            reward += info["completion_reward"]

        if completion_reward != 1. and self.stagnation_cost:
            if self._prev_reward is not None:
                if np.array_equal(self._prev_reward.shape, individual_reward.shape):
                    if np.array_equal(self._prev_reward, individual_reward):
                        info["stagnation_cost"] = self.stagnation_cost
                        reward -= info["stagnation_cost"]

            self._prev_reward = individual_reward

        return reward, done, info
