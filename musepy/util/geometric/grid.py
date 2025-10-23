from mindcraft.io import Repr
from numpy import array, asarray, atleast_2d, clip, eye, ndarray, sqrt, zeros, mod, concatenate, rint, empty, shape, all
from typing import Optional, Union
from mindcraft.util.geometric import transform_to_labframe, transform_to_coords
from copy import copy


class Grid(Repr):
    REPR_FIELDS = ("unit_cell", "size", "bounds", "rigid")
    PBC = "pbc"
    """ periodic boundary condition identifier """
    FBC = "fbc"
    """ fixed boundary condition identifier """
    NBC = "no"
    """ no boundary condition identifier """
    BOUNDARY_CONDITIONS = (PBC, FBC, NBC, None)

    def __init__(self, neighbors, unit_cell=None, size: int = 5, bounds: Optional[str] = FBC,
                 rigid: bool = True, **kwargs):
        Repr.__init__(self, to_list=("unit_cell", "size"), repr_fields=self.REPR_FIELDS, **kwargs)
        self.neighbors = asarray(neighbors, dtype=int)
        self.unit_cell = atleast_2d(unit_cell)
        self._size = None
        self.size = size
        self.pbc_low = zeros(len(self.size))
        assert bounds in self.BOUNDARY_CONDITIONS
        self.bounds = bounds
        self.rigid = rigid

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        if isinstance(value, int):
            value = tuple([value] * self.unit_cell.ndim)
        self._size = asarray(value)

    @property
    def ndim(self):
        return shape(self.neighbors)[-1]

    @classmethod
    def default(cls):
        return SquareGrid()

    @property
    def num_neighbors(self):
        return len(self.neighbors)

    def get_neighbors(self, coords: Optional[Union[ndarray, list, tuple]] = None, apply_bounds=True):
        """ Retrieves the immediate neighbors of `coords` (without the coords) of for the `Grid`
         as defined in the `neighbors` property. If a `coords` is/are defined, the immediate neighborhood of that
         particular coords is returned. """
        if coords is None or not len(coords):
            neighbors = self.neighbors.copy()

        else:
            coords = asarray(coords)
            if coords.ndim == 1:
                neighbors = coords + self.neighbors

            elif coords.ndim == 2:
                # return [coord_i + self.neighs, ...] as np.array
                neighbors = (coords + self.neighbors[:, None, :]).transpose(1, 0, 2)

            else:
                raise NotImplementedError(f"coord-dim {coords.ndim} not in (0, 1, 2).")

        if not apply_bounds:
            return neighbors

        return self.apply_bounds(neighbors)

    @property
    def neighborhood_size(self):
        return self.num_neighbors + 1

    def get_neighborhood(self, coords: Optional[Union[ndarray, list, tuple]] = None, apply_bounds=True):
        """ Retrieves the immediate neighborhood coordinates (with leading `coords`) for given `coords` on the `Grid`
         as defined in the `neighbors` property. If a `coords` is/are defined, the immediate neighborhood of that
         particular coords is returned. """
        if coords is None:
            neighborhood = concatenate([zeros((1, self.ndim), dtype=int), self.neighbors], axis=0)

        elif len(coords) == 0:
            return empty((0, self.num_neighbors, self.ndim))

        else:
            coords = asarray(coords)
            if coords.ndim == 1:
                neighborhood = concatenate([coords[None, :], coords + self.neighbors], axis=0)

            elif coords.ndim == 2:
                # return [coord_i + self.neighs, ...] as np.array
                neighborhood = (coords + self.neighbors[:, None, :]).transpose(1, 0, 2)
                neighborhood = concatenate([coords[:, None, :], neighborhood], axis=1)

            else:
                raise NotImplementedError(f"coord-dim {coords.ndim} not in (0, 1, 2).")

        if not apply_bounds:
            return neighborhood

        return self.apply_bounds(neighborhood)

    def to_labframe(self, coords):
        return atleast_2d(transform_to_labframe(coords, self.unit_cell))

    def to_coords(self, labframe_coords, apply_bounds=False):
        c = atleast_2d(transform_to_coords(labframe_coords, self.unit_cell))
        if not apply_bounds:
            return c

        return self.apply_bounds(c)

    @property
    def pbc(self):
        """ flag showing whether grid has **periodic** boundary conditions """
        return self.bounds == self.PBC

    @property
    def fbc(self):
        """ flag showing whether grid has **fixed** boundary conditions """
        return self.bounds == self.FBC

    @property
    def nbc(self):
        """ flag showing whether grid has **no** boundary conditions """
        return not self.pbc and not self.fbc

    def apply_bounds(self, coords):
        if all(shape(coords)):
            if self.pbc:
                return rint(mod(coords, self.size)).astype(int)

            elif self.fbc:
                return rint(clip(coords, 0., self.size - 1)).astype(int)

        return coords

    @property
    def is_square_grid(self):
        return isinstance(self, SquareGrid)

    @property
    def is_edge_square_grid(self):
        return isinstance(self, EdgeSquareGrid)

    @property
    def is_hexagonal_grid(self):
        return isinstance(self, HexGrid)


class LRSquareGrid(Grid):
    NEIGHBORS = array(((1, 0), (-1, 0)))
    UNIT_CELL = eye(2)

    def __init__(self, **kwargs):
        kwargs["neighbors"] = self.NEIGHBORS
        kwargs["unit_cell"] = self.UNIT_CELL
        Grid.__init__(self, **kwargs)


class LSquareGrid(Grid):
    NEIGHBORS = array(((-1, 0)))
    UNIT_CELL = eye(2)

    def __init__(self, **kwargs):
        kwargs["neighbors"] = self.NEIGHBORS
        kwargs["unit_cell"] = self.UNIT_CELL
        Grid.__init__(self, **kwargs)


class EdgeSquareGrid(Grid):
    NEIGHBORS = array(((1, 0), (0, 1), (-1, 0), (0, -1)))
    UNIT_CELL = eye(2)

    def __init__(self, **kwargs):
        kwargs["neighbors"] = self.NEIGHBORS
        kwargs["unit_cell"] = self.UNIT_CELL
        Grid.__init__(self, **kwargs)


class SquareGrid(Grid):
    NEIGHBORS = array(((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)))
    UNIT_CELL = eye(2)

    def __init__(self, **kwargs):
        kwargs["neighbors"] = self.NEIGHBORS
        kwargs["unit_cell"] = self.UNIT_CELL
        Grid.__init__(self, **kwargs)


class CubeGrid(Grid):
    NEIGHBORS = array(((1, 0, 0), (1, 1, 0), (0, 1, 0), (-1, 1, 0), (-1, 0, 0), (-1, -1, 0), (0, -1, 0), (1, -1, 0),
                       (1, 0, 1), (1, 1, 1), (0, 1, 1), (-1, 1, 1), (-1, 0, 1), (-1, -1, 1), (0, -1, 1), (1, -1, 1),
                       (1, 0, -1), (1, 1, -1), (0, 1, -1), (-1, 1, -1), (-1, 0, -1), (-1, -1, -1), (0, -1, -1), (1, -1, -1),
                       ))
    UNIT_CELL = eye(3)

    def __init__(self, **kwargs):
        kwargs["neighbors"] = self.NEIGHBORS
        kwargs["unit_cell"] = self.UNIT_CELL
        Grid.__init__(self, **kwargs)


class HexGrid(Grid):
    NEIGHBORS = array(((1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)))
    UNIT_CELL = array([[1., 0.], [0.5, 0.5 * sqrt(3)]])

    def __init__(self, **kwargs):
        kwargs["neighbors"] = self.NEIGHBORS
        kwargs["unit_cell"] = self.UNIT_CELL
        Grid.__init__(self, **kwargs)
