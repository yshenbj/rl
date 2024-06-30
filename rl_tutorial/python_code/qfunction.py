from tabular_policy import TabularPolicy


class QFunction:

    """ Update the Q-value of (state, action) by delta """

    def update(self, state, action, delta):
        abstract

    """ Get a Q value for a given state-action pair """

    def get_q_value(self, state, action):
        abstract

    """ Save a policy to a specified filename """
    def save_policy(self, filename):
        abstract

    """ Load a policy from a specified filename """
    def load_policy(self, filename):
        abstract

    """ Return the action with the maximum Q-value """
    def get_argmax_q(self, state, actions):
        (argmax_q, max_q) = self.get_max_pair(state, actions)
        return argmax_q

    """ Return the maximum Q-value in this Q-function """
    def get_max_q(self, state, actions):
        (argmax_q, max_q) = self.get_max_pair(state, actions)
        return max_q

    """ Return a pair containing the action and Q-value, where the
        action has the maximum Q-value in state
    """
    def get_max_pair(self, state, actions):
        arg_max_q = None
        max_q = float("-inf")
        for action in actions:
            value = self.get_q_value(state, action)
            if max_q < value:
                arg_max_q = action
                max_q = value
        return (arg_max_q, max_q)
