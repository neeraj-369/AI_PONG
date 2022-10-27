# import gym
# import time

# env = gym.make("MsPacman-v0")
# state = env.reset()


# env.render()
# l = []
# for _ in range(1000):
#     env.render()
#     action = env.action_space.sample()
#     state, reward, done, info = env.step(action)
#     l.append(action)
#     if(done):
#         state = env.reset()
#     # time.sleep(0.1)
# env.close()

# for i in l:
#     print(i)


import gym

env = gym.make('ma_gym:PongDuel-v0')
done = [False for _ in range(env.n_agents)]
ep_reward = 0

print(done)
observation = env.reset()
r = []
for _ in range(1000):
    env.render()
    action = env.action_space.sample()
    # print(action)
    observation, reward, done, info = env.step(action)
    ep_reward += sum(reward)
    r.append(action)

for i in r:
    print(type(i[0]))
env.close()