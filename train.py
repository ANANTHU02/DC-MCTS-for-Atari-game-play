!pip install gym[atari]
!pip install autorom[accept-rom-license]

import gym
import torch
import numpy as np

class Environment:
    def __init__(self, render_mode='human'):
        self.env = gym.make("PongNoFrameskip-v4")
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.state = self.env.reset()
        self.render_mode = render_mode

    def step(self, action):
        if action is not None:
            action = int(action)
        else:
            action = self.env.action_space.sample()
        next_state, reward, done, _ = self.env.step(action)
        self.state = next_state
        return next_state, reward, done, {}

    def reset(self):
        self.state = self.env.reset()
        return self.state

    def render(self):
        self.env.render(mode=self.render_mode)


class MCTS:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self.Q = {}
        self.N = {}
        self.C = 1.0

    def select_action(self, state):
        state_hashable = state.tobytes()
        if state_hashable not in self.Q:
            self.Q[state_hashable] = {}
            self.N[state_hashable] = {}

        ball_position = np.max(state[54])  # Maximum vertical position of the ball

        # Divide the search space based on the ball's vertical position
        if ball_position < 132:  # Ball is below the paddle
            subgame = 'below'
        else:  # Ball is above the paddle
            subgame = 'above'

        # Getting the subgame-specific Q and N dictionaries
        subgame_Q = self.Q[state_hashable].get(subgame, {})
        subgame_N = self.N[state_hashable].get(subgame, {})

        # Initializing the Q and N dictionaries for the subgame
        for action in range(self.env.action_space.n):
            if action not in subgame_Q:
                subgame_Q[action] = 0
                subgame_N[action] = 0

        total_visits = sum(subgame_N.values())
        if total_visits == 0:
            ucb_value = float('inf')
        else:
            ucb_value = subgame_Q[action] + self.C * np.sqrt(np.log(total_visits) / (subgame_N[action] + 1e-8))

        best_action = np.argmax(ucb_value)
        return best_action, subgame


    def backpropagate(self, state, action, reward, subgame):
        state_hashable = state.tobytes()

        subgame_Q = self.Q[state_hashable].get(subgame, {})
        subgame_N = self.N[state_hashable].get(subgame, {})

        if action not in subgame_Q:
            subgame_Q[action] = reward
            subgame_N[action] = 1
        else:
            subgame_Q[action] = ((subgame_Q[action] * subgame_N[action]) + reward) / (subgame_N[action] + 1)
            subgame_N[action] += 1

        # Update the Q and N dictionaries for the subgame
        self.Q[state_hashable][subgame] = subgame_Q
        self.N[state_hashable][subgame] = subgame_N


class Agent:
    def __init__(self, env):
        self.env = env
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, env.action_space.n),
        )

        self.mcts = MCTS(env, self.policy)

    def select_action(self, state):
        action, subgame = self.mcts.select_action(state)
        return action, subgame

    def update(self, state, action, reward, subgame):
        self.mcts.backpropagate(state, action, reward, subgame)


def train(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, subgame = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, subgame)
            state = next_state
            total_reward += reward

        print("Episode {}: Total Reward = {}".format(episode + 1, total_reward))


if __name__ == "__main__":
    env = Environment()
    agent = Agent(env)
    train(env, agent, 50)

    print("Training complete.")

torch.save(agent.policy.state_dict(), "trained_agent.pth")
