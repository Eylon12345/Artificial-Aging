from mindcraft import Env
from musepy.util.geometric import Grid
from musepy.util.state import State
from musepy.corpus import Task
from musepy.visualize import Render
import numpy as np


class HybridNCA(Env):
    REPR_FIELDS = ("grid", "task", "state", "incremental_reward", "fbc_bounds_clip", "render_on_reset",
                   "permit_action_init", "render",
                   *Env.REPR_FIELDS)

    def __init__(self,
                 grid: Grid,
                 task: Task,
                 state: State,
                 incremental_reward: bool = True,
                 fbc_bounds_clip=False,
                 render_on_reset=False,
                 permit_action_init=True,
                 render=None,
                 **env_kwargs,
                 ):
        Env.__init__(self, to_list=(), render_on_reset=render_on_reset, **env_kwargs)

        self.grid = Grid.make(grid)
        self.task = Task.make(task)
        self.state = State.make(state)
        self.render = Render.make(render)

        self.incremental_reward = incremental_reward
        self.fbc_bounds_clip = fbc_bounds_clip

        # helpers
        self.permit_action_init = permit_action_init
        self._action_init = False
        self._cumulative_reward = 0.
        self.simulation_step = 0

        self.info = {}
        self.reset()

    @property
    def shape(self):
        return self.state.shape

    @property
    def n_cells(self):
        return int(np.product(self.shape[:-1]))

    @property
    def n_types(self):
        return self.state.n_type

    @property
    def n_hidden(self):
        return self.state.n_hidden

    @property
    def n_channels(self):
        return self.state.ndim

    @property
    def state_array(self):
        return self.state.array

    def reset(self, action: object = None) -> object:
        if isinstance(self.render, Render):
            self.render.reset()

        init_kwargs = {"bounds": self.state.bounds}
        init_task = self.task.init_task(**init_kwargs)

        for attr_name, attr_inits in init_task.items():
            attr = getattr(self, attr_name)
            for init_name, init_val in attr_inits.items():
                setattr(attr, init_name, init_val)

        self.state.init_array(grid=self.grid)

        observation, info = self.get_cell_neigh_states()
        self.simulation_step = 0
        self._cumulative_reward = 0.
        self._action_init = False
        info["reset"] = True
        self.info = info
        return observation, 0., False, info  # state, reward, done, info

    def step(self, action: np.ndarray):
        reset = self.permit_action_init and not self._action_init
        action = action.reshape(*self.shape)
        self.state.update(action, reset=reset)
        reward, done, reward_info = self.get_reward()
        state, state_info = self.get_cell_neigh_states()
        info = {"reset": False, **state_info, **reward_info}
        self.simulation_step += 1
        self._action_init = True
        self.info = info
        return state, reward, done, info

    def get_reward(self):
        reward, done, info = self.task.evaluate(env=self)

        if self.incremental_reward:
            individual_reward = np.sum(info.get("individual_reward", reward))

            reward -= self._cumulative_reward
            self._cumulative_reward = individual_reward
            # reward += info.get("completion_reward", 0.)
            # reward -= info.get("stagnation_cost", 0.)

        return reward, done, info

    def get_cell_coords(self):
        # return 2d coords of cells within grid
        coords = np.indices(self.shape[:-1]).reshape(2, -1).T
        return coords

    def get_cell_neigh_states(self, return_all=False):
        state_info = {}

        coords = self.get_cell_coords()
        if self.fbc_bounds_clip:
            # shape:   (num_cells, num_neighs, xy)
            # out of bounds neighbors are clipped to the boundary of the lattice,
            # e.g., [-1, 0] -> [0, 0], [-1, -1] -> [0, 0], [-1, 1] -> [0, 1], ...
            # thus, out of bounds neighbors are replaced by the cell itself, or its in-bound neighbors
            neigh_coords = self.grid.get_neighborhood(coords)  # apply_bounds == True
        else:
            # no bounds clipping, e.g., [-1, 0] -> [-1, 0], ...
            # below, we check whether neighbors are within grid-bounds and omit out-of-bounds neighbors
            neigh_coords = self.grid.get_neighborhood(coords, apply_bounds=False)

        shape = neigh_coords.shape  # shape:   (num_cells,  num_neighs, xy)
        neigh_coords = neigh_coords.reshape((shape[0] * shape[1], shape[2]))  # flatten: (num_cells x num_neighs, xy)
        neigh_states = np.zeros((neigh_coords.shape[0], self.state.ndim))  # shape:   (num_cells x num_neighs, feat.)

        # check whether neighbors are within grid-bounds
        in_bounds = np.all(neigh_coords >= 0., axis=1)
        for i in range(shape[2]):  # x, y, ...
            in_bounds &= neigh_coords[:, i] < self.grid.size[i]

        if not self.grid.pbc:
            neigh_coords = neigh_coords[in_bounds]
            neigh_states[in_bounds] = self.state.array[neigh_coords[:, 0], neigh_coords[:, 1]]
        else:
            out_of_bounds = ~in_bounds
            neigh_coords[out_of_bounds] = self.grid.apply_bounds(neigh_coords[out_of_bounds])
            neigh_states[:] = self.state.array[neigh_coords]

        neigh_states = neigh_states.reshape((shape[0], shape[1], self.state.ndim))  # (num_cells, num_neighs, feat.)
        return neigh_states, state_info
