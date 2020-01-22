import gym
import numpy as np
from ppo_model import PPOModel
env = gym.make("CartPole-v0")
env.reset()

ppo_model = PPOModel(num_states=4, num_actions=1, epochs=1)
for i_episode in range(1000):
    observation = env.reset()
    observs = []
    rewards = []
    val_preds = []
    actions = []
    means = []
    for t in range(100):
        env.render()
        observation = np.reshape(observation, (1, observation.shape[0]))
        action, val = ppo_model.next_action_and_value(observation)
        observs.append(observation)
        actions.append(action)
        val_preds.append(val)
        means.append(ppo_model.distribution.mean)
        action = action.numpy()[0]
        print(action)
        action  = 1 if action >= 0.5 else 0
        observation, reward, done, info = env.step(action)
        rewards.append(reward)
        if done:
                print("Episode finished after {} timesteps.".format(t+1))
                ep_dic = {
                    "observations" : observs,
                    "rewards" : rewards,
                    "values" : val_preds,
                    "actions" : actions,
                    "means" : means
                    }
                ppo_model.add_vtarg_and_adv(ep_dic)
                ppo_model.train(ep_dic)
                break
env.close()
