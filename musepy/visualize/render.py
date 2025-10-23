from mindcraft.io import Repr
from mindcraft.util.tensor import to_categorical
import numpy as np


class Render(Repr):
    REPR_FIELDS = ("track_ids", "ion", "show", "colors", "cell_size", "single_channel")

    def __init__(self, track_ids=(), ion=True, show=True, xylim=None, colors=(), palette=None, mca=None, cell_size=1., static_size=True,
                 plt=None, fig=None, ax=None, jupyter=False, single_channel=None):
        self.track_ids = track_ids
        self.ion = ion
        self.show = show
        self.xylim = xylim
        self.colors = colors
        self.palette = palette
        self.cell_size = cell_size
        self.static_size = static_size
        self.mca = mca
        self.single_channel = single_channel

        # helpers
        self._jupyter = jupyter
        self._display, self._update, self._output = None, None, None
        if jupyter:
            from IPython.display import display, update_display
            self._display = lambda fig: display(fig, display_id=True)
            self._update = lambda fig, display_id: update_display(fig, display_id=display_id)

        self._plt = plt
        self._fig = fig
        self._ax = ax
        self._dont_reset_plt = any((plt is not None, ax is not None, jupyter))

        self._xlim = None
        self._ylim = None
        self._plot_range = None
        self._polygons = None
        self.step = 0
        Repr.__init__(self, to_list=("track_ids", "xylim", "rgb"))

    def reset(self):
        if not self._dont_reset_plt:
            if self._plt is not None:
                self._plt.close('all')
            self._plt = None
            self._fig = None
            self._ax = None
        self._xlim = None
        self._ylim = None
        self._plot_range = None
        self._polygons = None
        if not self._jupyter:
            self._display, self._update, self._output = None, None, None
        self.step = 0

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._plt is not None:
            self._plt.close('all')

    def _clear_ax(self):
        if self._plt is not None:
            plt = self._plt
            fig = self._fig
            ax = self._ax
            ax.clear()

        else:
            import matplotlib.pyplot as plt
            fig, ax = plt.gcf(), plt.gca()
            if self.ion:
                plt.ion()
                self._plt = plt
                self._fig = fig
                self._ax = ax

        title = f"Multicellular Environment, Step {self.step}"
        plt.gcf().canvas.manager.set_window_title(f"{self.mca.grid.__class__.__name__} {title}")
        ax.axis('equal')
        return plt, fig, ax

    def _get_cell_shape(self):
        if self.mca.grid.is_square_grid or self.mca.is_edge_square_grid:
            num_vertices = 4
            orientation = np.radians(45)
            radius = np.sqrt(2.) * (self.cell_size * 0.5)  # distance from center to vertices

        elif self.mca.grid.is_hexagonal_grid:
            num_vertices = 6
            orientation = np.radians(120)
            radius = (self.cell_size * 0.5) * 2. / np.sqrt(3)  # distance from center to vertices

        else:
            raise NotImplementedError(f"grid_type '{self.mca.grid}' not supported.")

        return num_vertices, orientation, radius

    def _get_plot_range(self, ax):
        if self.static_size and self._plot_range is not None:
            return self._xlim, self._ylim, self._plot_range

        if self.xylim is None:
            x_offset = 1
            if self.mca.grid.is_hexagonal_grid:
                x_offset = int(self.mca.grid.size[1] / np.sqrt(3))
            xlim, ylim = [-2, self.mca.grid.size[0] + x_offset], [-1, self.mca.grid.size[1]]

        else:
            try:
                xlim, ylim = self.xylim
            except TypeError:
                xlim = [0., self.xylim]
                ylim = [0., self.xylim]

        # lab-frame positions for plotting
        plot_range = [(x, y) for x, y in zip(xlim, ylim)]
        self._xlim, self._ylim, self._plot_range = xlim, ylim, plot_range
        return self._xlim, self._ylim, self._plot_range

    def _get_polygons(self, plot_range):
        num_vertices, orientation, radius = self._get_cell_shape()
        if self.static_size and self._polygons is not None:
            polygons = self._polygons

        else:
            # coords-range for plotting
            plot_coords = np.rint(self.mca.grid.to_coords(plot_range))

            # make coordinate grid for centers of polygons (note, for skewed `unit_cell`s an offset is needed (todo)
            if self.mca.grid.is_square_grid or self.mca.is_edge_square_grid:
                X, Y = np.mgrid[slice(min(plot_coords[:, 0]) - 2, max(plot_coords[:, 0]) + 3, 1.),
                slice(min(plot_coords[:, 1]) - 2, max(plot_coords[:, 1]) + 3, 1.)]
                X, Y = X.flatten(), Y.flatten()
                xy_coords = np.vstack((X, Y)).T

            elif self.mca.grid.is_hexagonal_grid:
                X, Y = [], []
                y = np.arange(plot_coords[0, 1] - 1, plot_coords[1, 1] + 1)
                for i, y_i in enumerate(y):
                    x = np.arange(plot_coords[0, 0] - i // 2 - 1, plot_coords[1, 0] + (len(y) - i) // 2 + 2)
                    X.extend(x)
                    Y.extend([y_i] * len(x))

                X, Y = np.asarray(X)[:, np.newaxis], np.asarray(Y)[:, np.newaxis]
                xy_coords = np.hstack((X, Y))

            else:
                raise NotImplementedError(f"grid_type '{self.mca.grid}' not supported.")

            polygons = []
            from matplotlib.patches import RegularPolygon
            xy_pos = self.mca.grid.to_labframe(xy_coords)
            for ij, xy in zip(xy_coords, xy_pos):
                color = "lightgray"
                grid_size = self.mca.grid.size
                if grid_size is not None:
                    if all(ij >= 0) and np.all(ij < grid_size):
                        color = "gray"

                polygon = RegularPolygon(xy,
                                         numVertices=num_vertices, radius=radius, orientation=orientation,
                                         facecolor=color, alpha=0.3, edgecolor='white', zorder=0,
                                         linewidth=3.)
                polygons.append(polygon)

            self._polygons = polygons

        cells = []
        colors = np.zeros((self.mca.n_cells, 3))
        if self.colors:
            bounds = [self.mca.state.bounds for channel in self.colors]
            colors = [self.mca.state.get_val(channel=channel).reshape(self.mca.n_cells, -1) for channel in self.colors]

            if self.palette is None or np.shape(colors)[-1] == 1 or self.single_channel is not None:
                colors = [np.atleast_1d((ci - bi[0]) / (bi[1] - bi[0])) for ci, bi in zip(colors, bounds)]
                colors = np.concatenate(colors, axis=-1)
                if colors.ndim > 2:
                    colors = colors.reshape(-1, colors.shape[-1])

                if self.single_channel is not None:
                    colors = colors[:, self.single_channel][:, None]

            else:
                colors = np.asarray(colors)
                if colors.shape[-1] == len(self.palette):
                    colors = np.concatenate(colors, axis=-1)
                    colors = to_categorical(colors, classes=self.palette)
                else:
                    num_classes = len(self.palette)
                    colors = [np.atleast_1d(np.floor((ci / (bi[1] - bi[0]) + bi[0]) * (num_classes - 1))) for ci, bi in
                              zip(colors, bounds)]
                    colors = np.concatenate(colors, axis=-1)
                    colors = np.take(self.palette, colors.flatten().astype(int))

        colors = np.asarray(colors)
        if colors.ndim > 1 and colors.shape[-1] == 1:
            from matplotlib.cm import get_cmap, ScalarMappable
            from matplotlib.colors import Normalize
            try:
                cmap = get_cmap(list(self.colors.values())[0])
            except ValueError:
                cmap = get_cmap("magma")

            cmap = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
            # colors = np.ones((len(colors), 3)) * (1. - colors)  # black <-> white

            colors = cmap.to_rgba(x=colors, alpha=1.0)[..., :3]

        elif colors.ndim > 1 and colors.shape[-1] != 3:
            raise NotImplementedError(f"Table of {colors.shape} colors.")

        from matplotlib.patches import RegularPolygon
        for i, (xy, color) in enumerate(zip(self.mca.get_cell_coords(), colors)):
            polygon = RegularPolygon(xy,
                                     numVertices=num_vertices, radius=radius, orientation=orientation,
                                     facecolor=color, alpha=0.66, edgecolor='k', zorder=1, linewidth=2.)
            cells.append(polygon)

        return self._polygons, cells

    def _track_ids(self):
        # if self.track_ids:
        #     positions = self.mca.pos
        #     for track_id in track_ids:
        #         coords = self.coords[track_id]
        #         pos = positions[track_id]
        #         neighbor_coords = self.get_neighborhood(track_id)
        #         neighbor_pos = transform_to_labframe(neighbor_coords, self.unit_cell)
        #         for neigh, neigh_coords in zip(neighbor_pos, neighbor_coords):
        #             neigh_is_cell = contains(self.coords, neigh_coords)
        #             ax.plot(*np.stack((pos, neigh)).T, linestyle=":",
        #                     color="lightgreen" if neigh_is_cell else "lightgray")
        #             if neigh_is_cell:
        #                 ax.scatter(*neigh[:, None], facecolors='none', edgecolors='lightgreen', s=100)
        #
        #         ax.scatter(*pos[:, None], facecolors='none', edgecolors='lightgreen', s=100)
        pass

    def _show(self, plt, fig):
        if self.ion:
            plt.draw()

        if self._jupyter:
            if self.step == 0:
                self._output = self._display(fig)
            else:
                self._update(fig, self._output.display_id)

        if self.show:
            plt.show()

            if self.ion:
                plt.pause(0.0001)

    def __call__(self, ax_title=None):
        """ A method to render a Multicellular Automaton Environment `MuCA` instance using matplotlib. """
        plt, fig, ax = self._clear_ax()
        xlim, ylim, plot_range = self._get_plot_range(ax)
        polygons, cells = self._get_polygons(plot_range)
        [ax.add_patch(p) for p in polygons]
        [ax.add_patch(c) for c in cells]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        if ax_title:
            ax.set_title(ax_title)
        self._track_ids()
        self._show(plt, fig)
        self.step += 1
        return ax

