import torch

agent = Agent(env)
agent.policy.load_state_dict(torch.load("trained_agent.pth"))




def test(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0  
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                print("Episode {}: Total Reward = {}".format(episode + 1, total_reward))

num_episodes = 10

for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
test(env, agent, num_episodes)
