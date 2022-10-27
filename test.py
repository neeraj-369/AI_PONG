from re import L
import gym
import numpy as np
import time
import json

############### Activation functions ####################

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def relu(vector):
    vector[vector < 0] = 0
    return vector
    
    
################# Neural net #######################

def neural_net(observation_matrix, weights):
    # Compute the new hidden layer values and the new output layer values using the observation_matrix and weights 
    hidden_layer_values = np.dot(weights['1'], observation_matrix)
    hidden_layer_values = relu(hidden_layer_values)
    output_layer_values = np.dot(hidden_layer_values.T, weights['2'])
    output_layer_values = sigmoid(output_layer_values)
    return hidden_layer_values, output_layer_values

def Move_up_or_down(probability):
    random_value = np.random.uniform()
    if random_value < probability:
        # up in openai gym
        return 1
    else:
         # down in openai gym
        return 2

def compute_gradient(gradient_log_p, hidden_layer_values, observation_values, weights):
    delta_L = gradient_log_p
    dC_dw2 = np.dot(hidden_layer_values.T, delta_L).ravel()
    delta_l2 = np.outer(delta_L, weights['2'])
    delta_l2 = relu(delta_l2)
    dC_dw1 = np.dot(delta_l2.T, observation_values)
    return {
        '1': dC_dw1,
        '2': dC_dw2
    }

def weights_update(weights, expectation_g_squared, g_dict, decay_rate, learning_rate):
    epsilon = 1e-5
    for layer_name in weights.keys():
        g = g_dict[layer_name]
        expectation_g_squared[layer_name] = decay_rate * expectation_g_squared[layer_name] + (1 - decay_rate) * g**2
        weights[layer_name] += (learning_rate * g)/(np.sqrt(expectation_g_squared[layer_name] + epsilon))
        g_dict[layer_name] = np.zeros_like(weights[layer_name]) # reset batch gradient buffer


############### Reinforcement learning ################

def discount_rewards(rewards, gamma):
   # Actions you took 20 steps before the end result are less important to the overall result than an action you took a step ago. This implements that logic by discounting the reward on previous actions based on how long ago they were taken
    discounted_rewards = np.zeros_like(rewards)
    # print(type(discounted_rewards[0][0]))
    running_add = 0
    for t in reversed(range(0, rewards.size)):
        if rewards[t] != 0:
            running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = float(running_add)
    discounted_rewards = discounted_rewards.astype(float)
    # print(type((discounted_rewards[0][0])))
    return discounted_rewards

def discount_plus_rewards(gradient_log_p, episode_rewards, gamma):
    # discount the gradient with the normalized rewards 
    discounted_episode_rewards = discount_rewards(episode_rewards, gamma)
    # standardize the rewards to be unit normal (helps control the gradient estimator variance)
    discounted_episode_rewards -= np.mean(discounted_episode_rewards)
    discounted_episode_rewards /= np.std(discounted_episode_rewards)
    return gradient_log_p * discounted_episode_rewards


#################### The game  ##########################
def main():
    env = gym.make("ma_gym:PongDuel-v0")
    # env = gym.make("Pong-v0")
    observation = env.reset() # This gets us the image
    observation = observation[0] + observation[1]
    observation = np.array(observation)
    observation = observation.reshape(20, 1)
    # hyperparameters
    episode_number = 0
    total_episodes = 2
    batch_size = 10
    gamma = 0.99 # discount factor for reward
    decay_rate = 0.99
    num_hidden_layer_neurons = 200
    input_dimensions = 2*10
    learning_rate = 1e-4
    reward_sum = 0
    reward_sum_1 = 0
    running_reward = None
    prev_processed_observations = None

    f = open('player_1.json')
    data = json.load(f)
    f.close()

    ff= open('player_2.json')
    data_1 = json.load(ff)
    ff.close()

    weights = {
        '1': np.array(data['1']),
        '2': np.array(data['2'])
    }
    weights_1 = {
        '1': np.array(data_1['1']),
        '2': np.array(data_1['2'])
    }


    # To be used with rmsprop algorithm 
    expectation_g_squared = {}
    expectation_g_squared_1 = {}
    g_dict = {}
    g_dict_1= {}
    for layer_name in weights.keys():
        expectation_g_squared[layer_name] = np.zeros_like(weights[layer_name])
        g_dict[layer_name] = np.zeros_like(weights[layer_name])

    for layer_name in weights_1.keys():
        expectation_g_squared_1[layer_name] = np.zeros_like(weights_1[layer_name])
        g_dict_1[layer_name] = np.zeros_like(weights_1[layer_name])

    episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_gradient_log_ps_1, episode_rewards, episode_rewards_1 = [], [], [], [], [], []
    count = 0
    l = []
    while episode_number < total_episodes:
        
        if(episode_number > 2):
            break
        env.render()
        processed_observations = observation
        hidden_layer_values, up_probability = neural_net(processed_observations, weights)

        hidden_layer_values, up_probability_1 = neural_net(processed_observations, weights_1)

        episode_observations.append((processed_observations.T.tolist()[0]))
        episode_hidden_layer_values.append(hidden_layer_values)

        action = Move_up_or_down(up_probability)
        action_1 = Move_up_or_down(up_probability_1)
        observation, reward, done, info = env.step([action, action_1])


        observation = observation[0] + observation[1]
        observation = np.array(observation)
        observation = observation.reshape(20, 1)
        reward_sum += reward[0]
        reward_sum_1 += reward[1]

        episode_rewards.append(reward[0])
        episode_rewards_1.append(reward[1])

        # see here: http://cs231n.github.io/neural-networks-2/#losses
        fake_label = 1 if action == 2 else 0
        loss_function_gradient = fake_label - up_probability
        episode_gradient_log_ps.append(loss_function_gradient)

        fake_label_1 = 1 if action_1 == 2 else 0
        loss_function_gradient_1 = fake_label_1 - up_probability_1
        episode_gradient_log_ps_1.append(loss_function_gradient_1)
        time.sleep(0.1)
        l.append(done)
        if done[0]: # an episode finished
            # count+=1
            episode_number += 1
            # Combine the following values for the episode
            episode_hidden_layer_values = np.array(episode_hidden_layer_values)
            episode_observations = np.array(episode_observations)
            episode_gradient_log_ps = np.vstack(episode_gradient_log_ps)
            episode_rewards = np.vstack(episode_rewards)


            episode_gradient_log_ps_1 = np.vstack(episode_gradient_log_ps_1)
            episode_rewards_1 = np.vstack(episode_rewards_1)


            # Tweak the gradient of the log_ps based on the discounted rewards
            episode_gradient_log_ps_discounted = discount_plus_rewards(episode_gradient_log_ps, episode_rewards, gamma)

            episode_gradient_log_ps_discounted_1 = discount_plus_rewards(episode_gradient_log_ps_1, episode_rewards_1, gamma)
            
            gradient = compute_gradient(
              episode_gradient_log_ps_discounted,
              episode_hidden_layer_values,
              episode_observations,
              weights
            )

            gradient_1 = compute_gradient(
              episode_gradient_log_ps_discounted_1,
              episode_hidden_layer_values,
              episode_observations,
              weights_1
            )

            # Sum the gradient for use when we hit the batch size
            for layer_name in gradient:
                g_dict[layer_name] += gradient[layer_name]

            for layer_name in gradient_1:
                g_dict_1[layer_name] += gradient_1[layer_name]

            if episode_number % batch_size == 0:
                weights_update(weights, expectation_g_squared, g_dict, decay_rate, learning_rate)
                weights_update(weights_1, expectation_g_squared_1, g_dict_1, decay_rate, learning_rate)


            episode_hidden_layer_values, episode_observations, episode_gradient_log_ps, episode_rewards, episode_gradient_log_ps_1, episode_rewards_1 = [], [], [], [], [], [] # reset values
            observation = env.reset() # reset env
            observation = observation[0] + observation[1]
            observation = np.array(observation)
            observation = observation.reshape(20, 1)
            processed_observations = observation
            reward_sum = 0
            reward_sum_1 = 0
            prev_processed_observations = None

    for i in l:
        print(i)
    print(len(l))
main()
