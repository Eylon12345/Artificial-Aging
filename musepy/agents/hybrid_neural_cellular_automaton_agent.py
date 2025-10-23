import numpy as np
import torch
from mindcraft.agents import TorchAgent
from mindcraft.torch.module import StateEmbedding
from mindcraft.torch.module import Embedding
from mindcraft.torch.module import Patchwork
from torch import tensor, Tensor, zeros, cat, transpose, isfinite
from numpy import shape, asarray, ndarray, rint
from numpy import all as np_all
from numpy import sum as np_sum
from typing import Union, Optional
from mindcraft.io.spaces import space_clip
from mindcraft.torch.util import tensor_to_numpy, get_closest


class HybridNCAAgent(TorchAgent):
    REPR_FIELDS = ('state_module', 'sensory_module', 'aggregate', 'sensitivity',
                   'sensitivity_reduction_rate', 'sensitivity_susceptibility',
                   *TorchAgent.REPR_FIELDS)

    MODELS = ('state_module', 'sensory_module', 
              *TorchAgent.MODULES)

    AGGREGATE_FLATTEN = 'flatten'
    AGGREGATE_MEAN = 'mean'
    AGGREGATE = (AGGREGATE_MEAN, AGGREGATE_FLATTEN)

    def __init__(self,
                 policy_module: Union[Patchwork, str, dict],
                 state_module: StateEmbedding = None,
                 sensory_module: Optional[Union[Embedding, str, dict]] = None,
                 aggregate: str = AGGREGATE_FLATTEN,
                 sensitivity: float = None,
                 sensitivity_reduction_rate: float = None,
                 sensitivity_susceptibility: float = 1.0,
                 **kwargs):
        """ Constructs a NCAAgent instance

        :param policy_module: A `Patchwork` module representation (either a `mindcraft.torch.module.MindModule`, 
                             `str` or `dict`) which outputs the final action of the `NCAAgent`.
        :param state_module: A `StateEmbedding` module instance, (dict-like) representation (or path), 
                             specifying a parameter list for initial states of the `NCAAgent`.
        :param sensory_module: (Optional) sensory module instance, (dict-like) representation
                               or path to an `Embedding` module, defaults to None.
        :param aggregate: Functionality to aggregate sensory preprocessing, choice of any in `AGGREGATE`, defaults to
                          `AGGREGATE_MEAN`.
        :param sensitivity: Sensitivity of each cell of the agent to sensory input, defaults to 1 (all channels are
                            open).
        :param sensitivity_reduction_rate: Rate at which the sensitivity of each cell is reduced, if the cell's actual
                                           neighbor-sensitivity (measured as open sensor_sensitivity channels)
                                           is higher than the cell's sensitivity, defaults to None.
        :param sensitivity_susceptibility: Fraction of cells that can lose sensitivity to neighbors, defaults to 1.0.
                                           This is only relevant if `sensitivity` is not None, and allows to model
                                           the loss of gap-junctions (GS) between a certain fraction of cells in the
                                           tissue.
        :param clip:  boolean, specifying whether the action is clipped to the allowed action space values,
                      defaults to True. If clip is set to 'forward' and the state_module is present and the
                      info-dict contains a 'reset' flag, the state_module's state is returned instead of the
                      clipped action.
        :param kwargs: Keyword arguments forwarded to the `TorchAgent` super-class.
        """
        self.state_module = None
        self.sensory_module = None
        self.embedding = None
        self.sensitivity = sensitivity
        self.sensitivity_reduction_rate = sensitivity_reduction_rate
        self.sensitivity_susceptibility = sensitivity_susceptibility
        self._non_susceptible = None

        modules = kwargs.pop("modules", {})
        if state_module is not None:
            modules['state_module'] = StateEmbedding.make(state_module)
        if sensory_module is not None:
            modules['sensory_module'] = Patchwork.make(sensory_module)

        TorchAgent.__init__(self, policy_module=policy_module, modules=modules, **kwargs)

        assert aggregate in self.AGGREGATE
        self.aggregate = aggregate
        self.sensory_memory = {}
        self.policy_memory = {}
        self.sensor_sensitivity = 1.

    def to_dict(self):
        dict_repr = TorchAgent.to_dict(self)
        return dict_repr

    def reset(self, observation, reward, info):
        self.embedding = None
        self.sensory_memory = {}
        self.policy_memory = {}
        return TorchAgent.reset(self, observation, reward, info)

    def to(self, device: str):
        """ move model (and submodels) to specified `device` (cpu or cuda)

            :returns: self
        """
        # move tensors
        if self.embedding is not None:
            self.embedding = self.embedding.to(device)

        return TorchAgent.to(self, device)  # moves modules and super-class tensors to device

    @property
    def hidden_state_policy_module(self):
        return self.hidden_state

    @property
    def hidden_state_sensory_module(self):
        try:
            hidden_state = self.sensory_module.hidden_state
            return hidden_state

        except AttributeError:
            return None

    def preprocess_observation(self, observation, info):
        # load cell_ids either from info-dict or via observation batch-size
        cell_ids = (info or {}).get('cell_ids', list(range(len(observation))))
        sensitivity = (info or {}).get('sensitivity', self.sensitivity)

        if observation is None or np_all(asarray(observation) == None):
            observation = self.get_default_state(observation, cell_ids)

        if not isinstance(observation, Tensor):
            observation = tensor(observation, device=self.device, dtype=self.dtype)

        # potentially move to correct device
        observation = observation.to(self.device)
        return self.mask_observation(observation), cell_ids, sensitivity

    def get_default_state(self, observation, cell_ids) -> object:
        if self.state_module is not None:
            default_state = self.state_module.state
            if default_state is not None:
                default_state = default_state.clone()

            else:
                coords = torch.meshgrid(*(torch.arange(1, n + 1) for n in self.state_module.state_size))
                default_state = self.state_module(coords)

            if default_state.shape[0] != 1:
                raise NotImplementedError(f"No multiple genomes allowed yet (used  {default_state.shape[0]}).")

            observation_shape = shape(observation)
            observation = zeros(*observation_shape, self.state_module.state_size)
            for i in range(len(observation)):
                observation[i, 0] = default_state.flatten()
        return observation

    def forward(self, observation: Optional[Union[ndarray, Tensor]], reward=None, info=None) -> object:
        """ evaluate the agent's action from an `observation` of the environment by applying the agent's

        and clip the result to the agent's action space boundaries (possible gradients are not affected).

        :param observation: observation state of the environment
        :param reward: reward signal from the environment.
        :param info: Optional info dictionary signal from the environment.
        :returns: proposed action, clipped to agent's action space boundaries (if `clip` is set),
                  as `numpy.ndarray` if the `np_fields` is set, as `Tensor` otherwise.
                  If only one observation is provided (i.e., if the batch-size is 1), the
                  return value is flattened to the action-space shape.
        """
        if info and info.get("reset") and self.state_module is not None:
            return self.forward_state_module()

        observation, cell_ids, sensitivity = self.preprocess_observation(observation, info)
        # reward = self.reward_to_tensors(reward)

        embedding = self.forward_sensory_module(observation, cell_ids, sensitivity)
        action = self.forward_policy_module(embedding, cell_ids)
        return self.forward_action(action)

    def forward_sensory_module(self, x, cell_ids, sensitivity, reshape=True):
        batch_size, neighbor_size, num_channels = x.shape
        x = self.apply_sensor_sensitivity(x, cell_ids, sensitivity)

        if self.sensory_module is None:
            embedding = x

        elif not self.sensory_module.is_sequence_module:
            embedding = self.sensory_module(x)

        else:
            # align previously evaluated hidden state
            if self.sensory_module.states is not None:
                memory_cell_state_indices, num_new_cells = self.get_sensory_memory(cell_ids)
                states = [s[:, memory_cell_state_indices] for s in self.sensory_module.states]
                if num_new_cells:
                    for i in range(len(states)):
                        new_state = zeros((states[i].shape[0], num_new_cells, *states[i].shape[2:]))
                        states[i] = cat([states[i], new_state], dim=1)

                if reshape:
                    self.sensory_module.states = self.reshape_sensory_cell_states(self.sensory_module,
                                                                                  batch_size=batch_size,
                                                                                  neighbor_size=neighbor_size,
                                                                                  states=states,
                                                                                  squeeze=True)

            # transform cell observations to independent batches
            if not self.sensory_module.aggregates(dim=1):
                x = x.reshape(batch_size * neighbor_size, num_channels)

            # forward model
            embedding = self.sensory_module(x)

            if reshape and not self.sensory_module.aggregates(dim=1):
                single_embedding = not isinstance(embedding, tuple)
                if single_embedding:
                    embedding = (embedding,)
                embedding = tuple([e.reshape((batch_size, neighbor_size, e.shape[-1])) for e in embedding])
                if single_embedding:
                    embedding = embedding[0]

            # reshape all hidden state to (num_layers, batch_size, num_cells, features)
            self.set_sensory_memory(cell_ids)
            self.sensory_module.states = self.reshape_sensory_cell_states(self.sensory_module,
                                                                          batch_size=batch_size,
                                                                          neighbor_size=neighbor_size,
                                                                          squeeze=False)

        embedding = self.forward_aggregate(embedding, batch_size)
        return embedding

    def apply_sensor_sensitivity(self, x, cell_ids, sensitivity):
        """ open or block cell neighbor inputs based on the `sensitivity` of the cell """
        if sensitivity is None:
            return x

        batch_size, neighbor_size, num_channels = x.shape
        if isinstance(self.sensor_sensitivity, float):
            # initialize sensor_sensitivity buffer, open all sensors initially
            self.sensor_sensitivity = torch.ones((batch_size, neighbor_size), device=self.device, dtype=self.dtype)

        # remove cells in the sensor_sensitivity buffer that are not in the cell_ids anymore (may happen by cell-death)
        memory_cell_state_indices, num_new_cells = self.get_sensory_memory(cell_ids)
        self.sensor_sensitivity = self.sensor_sensitivity[memory_cell_state_indices]
        if num_new_cells:
            new_sensory_mask = torch.ones((num_new_cells, neighbor_size), device=self.device, dtype=self.dtype)
            self.sensor_sensitivity = cat([self.sensor_sensitivity, new_sensory_mask], dim=0)

        if isinstance(sensitivity, float):
            # initialize sensitivity buffer, set all cells to the same sensitivity if global float is provided
            sensitivity = torch.tensor([sensitivity] * batch_size, device=self.device, dtype=self.dtype)

        # set cell sensitivity to the closest possible sensitivity value that is possible with the given neighbor_size
        possible_sensitivity = [li / neighbor_size for li in range(neighbor_size + 1) if li != 0]
        sensitivity = get_closest(torch.clip(sensitivity, 0., 1.), possible_sensitivity)

        # get the number of open sensors for each cell, and evaluate whether to open or close sensors
        open_sensors = self.sensor_sensitivity.sum(axis=1)
        reduce_sensitivity = torch.where((open_sensors / neighbor_size) > sensitivity)[0]
        increase_sensitivity = torch.where((open_sensors / neighbor_size) < sensitivity)[0]

        # block additional cell neighbor inputs if open_sensors is too high
        for i in reduce_sensitivity:
            if self.sensitivity_reduction_rate is not None and torch.rand(1) < self.sensitivity_reduction_rate:
                continue

            s, num_blocked, mask = 1. - sensitivity[i], neighbor_size - open_sensors[i], self.sensor_sensitivity[i]
            should_be_blocked = rint(len(mask) * s)
            while neighbor_size - mask.sum() < should_be_blocked:
                open_neighbors = torch.where(mask)[0]
                mask[open_neighbors[torch.randint(1, len(open_neighbors), (1,))]] = 0.

        # open closed cell neighbor inputs if open_sensors is too low
        for i in increase_sensitivity:
            s, num_open, mask = sensitivity[i], open_sensors[i], self.sensor_sensitivity[i]
            should_be_open = rint(len(mask) * s)
            while mask.sum() < should_be_open:
                closed_neighbors = torch.where(mask == 0.)[0]
                mask[closed_neighbors[torch.randint(1, len(closed_neighbors), (1,))]] = 1.

        if self.sensitivity_susceptibility != 1.:
            num_non_susceptible = int(np.rint((1. - self.sensitivity_susceptibility) * len(self.sensor_sensitivity)))
            if self._non_susceptible is None or len(self._non_susceptible) != num_non_susceptible:
                self._non_susceptible = torch.randint(len(self.sensor_sensitivity), (num_non_susceptible,))

            self.sensor_sensitivity[self._non_susceptible] = 1.

        # mask cell neighbor inputs
        return x * self.sensor_sensitivity[..., None]

    def update_sensitivity_mask(self, cell_id, neighbor_size):
        if self.sensitivity is None:
            return None

        return tensor([self.sensitivity[i] for i in self.sensor_sensitivity], device=self.device, dtype=self.dtype)

    @staticmethod
    def reshape_sensory_cell_states(module, batch_size, neighbor_size, states: Optional[Union[tuple, list]] = None, squeeze=False):
        """

        :param module:
        :param batch_size:
        :param neighbor_size:
        :param states: Optional states list or tuple, if not provided, the `modules.states` will be used.
        :param squeeze: Flag whether to squeeze the cell's states to `(num_layers, batch_size * neighbor_size, -1)`,
                        or to unsqueeze to `(num_layers, batch_size, neighbor_size, hidden_size)`
        :return:
        """
        states = states or module.states
        num_layers = module.num_layers
        num_layers = num_layers if isinstance(num_layers, tuple) else [num_layers] * len(states)
        if squeeze:
            # transform to shape (NUM_LAYERS, BATCH_SIZE x NUM_NEIGHBORS, HIDDEN_SIZE)
            states = [s.reshape((n, batch_size * neighbor_size, -1)) for n, s in zip(num_layers, states)]
        else:
            # transform to shape (NUM_LAYERS, BATCH_SIZE, NUM_NEIGHBORS, HIDDEN_SIZE)
            states = [s.reshape((n, batch_size, neighbor_size, -1)) for n, s in zip(num_layers, states)]
        return states

    def set_sensory_memory(self, cell_ids):
        self.sensory_memory = {cell_id: i for i, cell_id in enumerate(cell_ids)}

    def get_sensory_memory(self, cell_ids):
        if self.sensory_memory is None:
            self.sensory_memory = {}

        num_new_cells = np_sum([cell_id not in self.sensory_memory for cell_id in cell_ids])
        memory_cell_state_indices = [self.sensory_memory[i] for i in cell_ids if i in self.sensory_memory]
        return memory_cell_state_indices, num_new_cells

    def forward_aggregate(self, embedding: Union[Tensor, tuple], batch_size):
        single_embedding = not isinstance(embedding, tuple)
        if single_embedding:
            embedding = (embedding,)

        if self.aggregate == self.AGGREGATE_FLATTEN:
            embedding = tuple([e.reshape(batch_size, -1) for e in embedding])
        elif self.aggregate == self.AGGREGATE_MEAN:
            embedding = tuple([e.mean(dim=1) for e in embedding])
        else:
            raise NotImplementedError(self.aggregate)

        if single_embedding:
            embedding = embedding[0]
        self.embedding = embedding
        return embedding

    def set_policy_memory(self, cell_ids):
        self.policy_memory = {cell_id: i for i, cell_id in enumerate(cell_ids)}

    def get_policy_memory(self, cell_ids):
        if self.policy_memory is None:
            self.policy_memory = {}

        num_new_cells = np_sum([cell_id not in self.policy_memory for cell_id in cell_ids])
        memory_cell_state_indices = [self.policy_memory[i] for i in cell_ids if i in self.policy_memory]
        return memory_cell_state_indices, num_new_cells

    @property
    def keep_policy_memory(self):
        if isinstance(self._sampling_count, np.ndarray):
            return True

        if self.policy_module is None:
            return False

        return self.policy_module.is_sequence_module

    def forward_policy_module(self, x, cell_ids):
        if self.keep_policy_memory:
            memory_cell_state_indices, num_new_cells = self.get_policy_memory(cell_ids)
            if isinstance(self._sampling_count, np.ndarray):
                self._sampling_count = self._sampling_count[memory_cell_state_indices]
                if num_new_cells:
                    self._sampling_count = np.concatenate([self._sampling_count, np.ones(num_new_cells, dtype=int)])
            self.set_policy_memory(cell_ids)

        if self.policy_module is None:
            return x

        if not isinstance(x, tuple):
            x = (x,)

        if not self.policy_module.is_sequence_module:
            return self.policy_module(*x)

        # align previously evaluated hidden state
        if self.policy_module.states is not None:
            states = [s[:, memory_cell_state_indices] for s in self.policy_module.states]
            if num_new_cells:
                for i in range(len(states)):
                    # num_layers, new_cells, ...
                    new_state = zeros((states[i].shape[0], num_new_cells, *states[i].shape[2:]))
                    states[i] = cat([states[i], new_state], dim=1)
            self.policy_module.states = states

        # forward model
        action = self.policy_module(*x)
        assert isfinite(action).all(), action
        return action

    def forward_state_module(self):
        action = self.state_module.state
        if action is not None:
            action = action.clone()
        else:
            coords = torch.meshgrid(*(torch.arange(1, n + 1) for n in self.state_module.num_states))
            coords = torch.stack(coords, dim=-1)
            coords = coords.reshape(-1, coords.size(-1))
            action = self.state_module(coords)

        action = tensor_to_numpy(action) if self.np_fields else action.clone()
        if self.clip and not self.clip == 'forward':
            action = space_clip(action, self.action_space)
        return action
