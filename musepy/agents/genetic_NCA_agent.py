from .hybrid_neural_cellular_automaton_agent import HybridNCAAgent


class GeneNCAAgent(HybridNCAAgent):
    """
    DiscreteNCAAgent is a subclass of HybridNCAAgent that operates with a discrete version of neural cellular automaton parameters.
    The parameters are transformed into continuous ANN parameters, which are then used to evaluate the fitness of the agent.
    The agent works in a continuous state space but is designed to handle discrete actions or states.
    """

    REPR_FIELDS = ("num_letters", *HybridNCAAgent.REPR_FIELDS)

    def __init__(self, num_letters=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_letters = num_letters

    def _transform_params(self, params):
        """
        Transform the discrete parameters into a continuous space.
        """
        if self.num_letters % 2 == 0:
            transpose = self.num_letters // 2
        else:
            transpose = (self.num_letters - 1) // 2

        params = params - transpose  # Shift the range to be centered around zero
        transformed_params = 2 * params / (self.num_letters - 1)  # Normalize to the range [-1., 1.]

        return transformed_params

    def _inverse_transform_params(self, params):
        """
        Inverse transform the continuous parameters back to discrete space.
        """

        params = (params + 1) * (self.num_letters - 1) / 2  # Scale back to [0, num_letters - 1]
        return params

    def get_parameters(self, modules=None):
        """ get all parameters of the agent's `unfreeze`d modules as a single concatenated tensor.

        :param modules: Optional list of modules to get parameters from, defaults to `unfreeze`d modules.
        :returns: either a concatenated tensor of all parameters of the agent's `unfreeze`d modules,
                  or numpy array if `np_fields` is set.
        """
        p = super().get_parameters(modules)
        if p is None:
            return None

        return self._inverse_transform_params(p)

    def set_parameters(self, parameters, modules=None):
        """ set the parameters of the agent's `unfreeze`d modules from a single concatenated tensor."""
        p = self._transform_params(parameters)
        return super().set_parameters(p, modules)
