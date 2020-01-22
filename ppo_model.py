"Based on https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py"

import random, threading, json
import tensorflow as tf
import numpy as np
from normal_distribution import NormalDistribution


class PPOModel:
    def __init__(self,
            num_states,
            should_load_model=False,
            num_actions=6 ,
            hidden_size=64,
            num_hidden_layers = 2,
            epsilon_clip=0.1,
            gamma=0.99,
            lam=0.95,
            entropy_coeff=0.0,
            clip_param=0.1,
            epochs=5,
            use_conv = False):
        self.training_json = "training.json"
        self.num_states = num_states
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.lose_rate = 1e-4
        self.var = 1.0
        self.epsilon_clip = epsilon_clip
        self.distribution = NormalDistribution(num_actions=num_actions)
        self.use_conv = use_conv
        if should_load_model:
            self.load_models()
        else:
            self.build_actor_and_critic()
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.gamma = gamma
        self.lam = lam
        self.entropy_coeff = entropy_coeff
        self.epochs = epochs
        self.train_lock = threading.Lock()
    
    def ppo_loss_continuous(self, advantage, old_prediction):
        def loss(y_true, y_pred):
            denom = tf.keras.backend.sqrt(tf.keras.backend.variable(2.0 * np.pi * self.var))
            prob_num = tf.keras.backend.exp(- tf.keras.backend.square(y_true - y_pred) / (2 * self.var))
            old_prob_num = tf.keras.backend.exp(- tf.keras.backend.square(y_true - old_prediction) / (2 * self.var))

            prob = prob_num/denom
            old_prob = old_prob_num/denom
            
            r = prob/(old_prob + 1e-10)

            return -tf.keras.backend.mean(tf.keras.backend.minimum(r * advantage, tf.keras.backend.clip(r, min_value=1 - self.epsilon_clip, max_value=1 + self.epsilon_clip) * advantage))
        return loss

    
    def build_actor_and_critic(self):
        if self.use_conv:
            self.build_actor_conv()
            self.build_critic_conv()
            print("Building CNN.")
        else:
            self.build_actor()
            self.build_critic()
        self.optimizer = tf.keras.optimizers.Adam()
        self.actor.summary()
        self.critic.summary()
        self.training_info = {"episode" : 0}

    def build_actor(self):
        inputs = tf.keras.Input(shape=(self.num_states,))
        x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(inputs)
        for _ in range(self.num_hidden_layers - 1):
            x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(x)
        out_actor = tf.keras.layers.Dense(self.num_actions, kernel_initializer=tf.random_normal_initializer())(x)
        self.actor = tf.keras.models.Model(inputs=[inputs], outputs=[out_actor])

    def build_critic(self):
        inputs = tf.keras.Input(shape=(self.num_states,))
        x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(inputs)
        for _ in range(self.num_hidden_layers - 1):
            x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(x)
        out_critic = tf.keras.layers.Dense(1, kernel_initializer=tf.random_normal_initializer())(x)
        self.critic = tf.keras.models.Model(inputs=[inputs], outputs=[out_critic])

    def build_actor_conv(self):
        inputs = tf.keras.Input(shape=(self.num_states,1,1))
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), padding="VALID", activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), padding="VALID", activation="relu")(x)
        out_actor = tf.keras.layers.Dense(self.num_actions, activation="linear", kernel_initializer=tf.random_normal_initializer())(x)
        self.actor = tf.keras.models.Model(inputs=[inputs], outputs=[out_actor])

    def build_critic_conv(self):
        inputs = tf.keras.Input(shape=(self.num_states,1,1))
        x = tf.keras.layers.Conv2D(filters=16, kernel_size=(8,8), strides=(4,4), padding="VALID", activation="relu")(inputs)
        x = tf.keras.layers.Conv2D(filters=32, kernel_size=(4,4), strides=(2,2), padding="VALID", activation="relu")(x)
        out_critic = tf.keras.layers.Dense(1, activation="linear", kernel_initializer=tf.random_normal_initializer())(x)
        self.critic = tf.keras.models.Model(inputs=[inputs], outputs=[out_critic])


    def next_action_and_value(self, observ):
        self.distribution.mean = self.actor(observ)
        return self.distribution.sample(), self.critic(observ)
    

    def add_vtarg_and_adv(self, ep_dic):
        """
        Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        """
        T = len(ep_dic["rewards"])
        ep_dic["values"] = np.append(ep_dic["values"], 0)
        ep_dic["adv"] = gaelam = np.empty(T, 'float32')
        lastgaelam = 0
        for t in range(T-1, -1, -1):
            nonterminal = 0 if t == T-1 else 1
            delta = ep_dic["rewards"][t] + self.gamma * ep_dic["values"][t+1] * nonterminal- ep_dic["values"][t] # TD Error
            gaelam[t] = lastgaelam = delta + self.gamma * self.lam * lastgaelam * nonterminal
        ep_dic["values"] = np.delete(ep_dic["values"], -1)
        ep_dic["tdlamret"] = ep_dic["adv"] + ep_dic["values"]
        ep_dic["adv"] = (ep_dic["adv"] - ep_dic["adv"].mean()) / ep_dic["adv"].std()
    
    def add_ret_and_adv(self, ep_dic):
        T = len(ep_dic["rewards"])
        ep_dic["adv"] = np.empty(T, 'float32')
        ep_dic["returns"] = np.empty(T, 'float32')
        for t in range(T-1, -1, -1):
            ep_dic["returns"][t] = ep_dic["rewards"][t]
            if t < T-1:
                ep_dic["returns"][t] += ep_dic["returns"][t+1] * self.gamma
        ep_dic["adv"] = ep_dic["returns"] - ep_dic["values"]
        #ep_dic["adv"] = (ep_dic["adv"] - ep_dic["adv"].mean()) / ep_dic["adv"].std()

    def train(self, ep_dic):
        with self.train_lock:
            print("Training")
            print("Rewards: ", ep_dic["rewards"])
            print("tdlamret: ", ep_dic["tdlamret"])
            print("Values: ", ep_dic["values"])
            epoch_bonus = 0#5 if ep_dic["rewards"][-1] > 0 else 0 
            print("BONUS: ", epoch_bonus, " REWARD: ", ep_dic["rewards"][-1])
            self.shuffle_ep_dic(ep_dic)
            for _ in range(self.epochs + epoch_bonus):
                for i in range(len(ep_dic["observations"])):
                    with tf.GradientTape(persistent=True) as tape:
                        policy_loss = self.ppo_loss(ep_dic, i)
                    #print("VALUE LOSS: ", value_loss)
                    grads = tape.gradient(policy_loss, self.actor.trainable_variables)
                    #print("GRADS: ", grads)
                    self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
                    #grads = tape.gradient(value_loss, self.critic.trainable_variables)
                    #print("GRADS: ", grads)
                    #self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
                    del tape
            
            observ_arr = np.array(ep_dic["observations"])
            observ_arr = np.reshape(observ_arr, (observ_arr.shape[0], observ_arr.shape[2]))
            self.critic.fit(observ_arr, ep_dic["tdlamret"], batch_size=32, epochs=2)
            self.training_info["episode"] += 1
            self.save_models()
            print("Done Training")
            return self.actor.get_weights(), self.critic.get_weights()

    def shuffle_ep_dic(self, ep_dic):
        seed = random.random()
        for k in ep_dic:
            if isinstance(ep_dic[k], list):
                random.seed(seed)
                random.shuffle(ep_dic[k])

    def value_loss(self, value, ret):
        return tf.reduce_mean(tf.square(ret - value))

    def ppo_loss(self, ep_dic, index):
        _, cur_val = self.next_action_and_value(ep_dic["observations"][index])

        old_distribution = NormalDistribution(mean=ep_dic["means"][index])
        kl_divergence = self.distribution.kl(old_distribution)
        entropy = self.distribution.entropy()
        mean_kl = tf.reduce_mean(kl_divergence)
        mean_entropy = tf.reduce_mean(entropy)
        policy_entropy_pen = -self.entropy_coeff * mean_entropy
        ratio = tf.exp(self.distribution.logp(ep_dic["actions"][index]) - old_distribution.logp(ep_dic["actions"][index]))
        surrogate_1 = ratio * ep_dic["adv"][index]
        surrogate_2 = tf.clip_by_value(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * ep_dic["adv"][index]
        policy_surrogate = - tf.reduce_mean(tf.minimum(surrogate_1, surrogate_2))
        #value_fn_loss = tf.reduce_mean(tf.square(ep_dic["tdlamret"][index] - cur_val))
        
        #value_fn_loss = tf.reduce_mean(tf.square(tf.convert_to_tensor(ep_dic["returns"][index]) - cur_val[0][0]))

        total_loss = policy_surrogate + policy_entropy_pen 
        return total_loss#, value_fn_loss
    
    def save_models(self):
        if self.use_conv:
            self.actor.save("actor_model_conv.h5")
            self.critic.save("critic_model_conv.h5")
        else:
            self.actor.save("actor_model.h5")
            self.critic.save("critic_model.h5")
        with open(self.training_json, "w") as f:
            json.dump(self.training_info, f)

    def load_models(self):
        print("LOADING MODELS")
        if self.use_conv:
            self.actor = tf.keras.models.load_model("actor_model_conv.h5")
            self.critic = tf.keras.models.load_model("critic_model_conv.h5")
        else:
            self.actor = tf.keras.models.load_model("actor_model.h5")
            self.critic = tf.keras.models.load_model("critic_model.h5")
        self.optimizer = tf.keras.optimizers.Adam()
        self.actor.summary()
        self.critic.summary()

        with open(self.training_json, "r") as f:
            self.training_info = json.load(f)

    # def update_old_model(self):
    #     self.actor_old = tf.keras.models.clone_model(self.actor)
    #     self.actor_old.set_weights(self.actor.get_weights())

#ppo_model = PPOModel(10)
#ppo_model.build_actor_and_critic()
