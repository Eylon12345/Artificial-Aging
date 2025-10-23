from mindcraft.io import Repr
from typing import Optional, Union
from numpy import clip, random, zeros


class State(Repr):
    """ """
    REPR_FIELDS = ('channels', 'bounds', 'noise_level')

    TYPE_CHL = "type"
    HIDDEN_CHL = "hidden"
    REWARD_CHL = "reward"
    CHANNELS = (TYPE_CHL, HIDDEN_CHL, REWARD_CHL, )
    """ A tuple listing allowed channels `CHANNELS`. """

    def __init__(self,
                 channels: dict,
                 bounds: Optional[Union[list, float]] = None,
                 noise_level=0.0,
                 initialize=None,
                 **kwargs):
        """  """
        assert all(k in self.CHANNELS for k in channels.keys())
        self.channels = channels
        if all(isinstance(v, int) for v in self.channels.values()):
            # integers provided for size of channel, translate to {key: (idx-start, idx-stop)}
            c, i = {}, 0
            for k, v in channels.items():
                c[k] = (i, i + v)
                i += v
            self.channels = c

        self.bounds = bounds or []
        if not hasattr(self.bounds, "__iter__"):
            self.bounds = [-self.bounds, self.bounds]

        self.noise_level = noise_level
        self._array = None
        self.initialize = initialize
        Repr.__init__(self, **kwargs)

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        if self._array is not None:
            self._array[...] = value
        else:
            self._array  = value

    def to_dict(self):
        dict_repr = Repr.to_dict(self)
        for attr_key in ('channels', ):
            attr = dict_repr[attr_key]
            for k in attr.keys():
                attr[k] = list(attr[k])
        dict_repr["bounds"] = list(self.bounds)
        return dict_repr

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return max((i[1] for i in self.channels.values()))

    def channel_idx(self, channel, to_slice=False):
        idx = self.channels[channel]
        if not to_slice:
            return idx
        return slice(*idx)

    def get_val(self, channel=None):
        if channel is None:
            return self.array[..., :]

        return self.array[..., self.channel_idx(channel, to_slice=True)]

    def set_val(self, values, channel=None):
        if channel is None:
            self.array[..., :] = values

        else:
            self.array[..., self.channel_idx(channel, to_slice=True)] = values

        self.clip()

    def get_channel_size(self, channel):
        l, h = self.channels[channel]
        return h - l

    def get_scale(self, channel):
        return self.bounds[channel][1] - self.bounds[channel][0]

    @classmethod
    def is_type(cls, channel):
        return channel == cls.TYPE_CHL

    @classmethod
    def is_hidden(cls, channel):
        return channel == cls.HIDDEN_CHL

    @property
    def lower_bound(self):
        return self.bounds[0]

    @property
    def upper_bound(self):
        return self.bounds[0]

    @property
    def n_type(self):
        try:
            return self.get_channel_size(self.TYPE_CHL)
        except KeyError:
            return 0

    @property
    def n_hidden(self):
        try:
            return self.get_channel_size(self.HIDDEN_CHL)
        except KeyError:
            return 0

    @property
    def n_reward(self):
        try:
            return self.get_channel_size(self.REWARD_CHL)
        except KeyError:
            return 0

    def get_noise(self):
        noise = 0.
        if self.noise_level:
            noise = random.randn(*self.array.shape) * self.noise_level

        return noise

    def update(self, value, reset=False):
        noise = self.get_noise()
        if reset:
            self.set_val(value + noise)
        else:
            self.set_val(self.array + value + noise)

        return self.array

    def clip(self):
        # clip to state bounds
        self.array = clip(self.array, *self.bounds)

    def init_array(self, grid, values=None):
        if values is None:
            self.array = zeros((*grid.size, self.ndim))
            for channel, value in self.initialize.items():
                self.set_val(value, channel)

        else:
            self.array = values

        return self.array
