import numpy as np
from collections import defaultdict
import random

class Agent:

    def __init__(self, nA=6, eps=0.001, alpha=0.2, gamma=0.99):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma

    def update_Q_sarsamax(self, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0  # value of next state
        target = reward + (self.gamma * Qsa_next)  # construct TD target
        new_value = current + (self.alpha * (target - current))  # get updated value
        return new_value

    def update_Q_sarsa(self, state, action, reward, next_state=None, next_action=None):
        """Returns updated Q-value for the most recent experience."""
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        # get value of state, action pair at next time step
        Qsa_next = self.Q[next_state][next_action] if next_state is not None else 0
        target = reward + (self.gamma * Qsa_next)  # construct TD target
        new_value = current + (self.alpha * (target - current))  # get updated value
        return new_value

    def update_Q_expsarsa(self, state, action, reward, next_state=None):
        """Returns updated Q-value for the most recent experience."""
        current = self.Q[state][action]  # estimate in Q-table (for current state, action pair)
        policy_s = np.ones(self.nA) * self.eps / self.nA  # current policy (for next state S')
        policy_s[np.argmax(self.Q[next_state])] = 1 - self.eps + (self.eps / self.nA)  # greedy action
        Qsa_next = np.dot(self.Q[next_state], policy_s)  # get value of state at next time step
        target = reward + (self.gamma * Qsa_next)  # construct target
        new_value = current + (self.alpha * (target - current))  # get updated value
        return new_value

    def epsilon_greedy(self, state):
        """Selects epsilon-greedy action for supplied state.

        Params
        ======
            Q (dictionary): action-value function
            state (int): current state
            nA (int): number actions in the environment
            eps (float): epsilon
        """
        if random.random() > self.eps:  # select greedy action with probability epsilon
            return np.argmax(self.Q[state])
        else:  # otherwise, select an action randomly
            return random.choice(np.arange(self.nA))

    def select_action(self, state):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return self.epsilon_greedy(state)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """

        if not done:
            next_action = self.epsilon_greedy(next_state)  # epsilon-greedy action
            #self.Q[state][action] = self.update_Q_sarsamax(state, action, reward, next_state)
            self.Q[state][action] = self.update_Q_expsarsa(state, action, reward, next_state)

            state = next_state  # S <- S'
            action = next_action  # A <- A'
        if done:
            self.Q[state][action] = self.update_Q_sarsa(state, action, reward)
