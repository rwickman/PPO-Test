"Based on https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py"

import random, threading, json
import tensorflow as tf
import numpy as np
from normal_distribution import NormalDistribution
from keras import backend as K


var = 0.6
epsilon_clip = 0.1

def ppo_loss_continuous(advantage, old_pred):
        def loss(y_true, y_pred):
            # y_true is actions taken in episode
            # y_pred is new means predictions
            # old_pred is old mean predictions
            n = old_pred.shape[1]
            new_logp = - n/2 * K.log(2*np.pi) - (1/(2 * var)) * K.sum(K.square(y_true - y_pred), axis=-1)
            old_logp = - n/2 * K.log(2*np.pi) - (1/(2 * var)) * K.sum(K.square(y_true - old_pred), axis=-1)

            ratio = K.exp(new_logp - old_logp)
            surrogate_1 = ratio * advantage
            surrogate_2 = K.clip(ratio, 1.0 - epsilon_clip, 1.0 + epsilon_clip) * advantage

            return - K.mean(K.minimum(surrogate_1, surrogate_2))
        return loss



class PPOModel:
    def __init__(self,
            num_states,
            num_actions=6 ,
            should_load_models=False,
            hidden_size=64,
            num_hidden_layers = 2,
            ep_clip=0.1,
            gamma=0.99,
            lam=0.95,
            entropy_coeff=0.0,
            clip_param=0.1,
            epochs=5,
            batch_size=32,
            use_conv = False):
        self.training_json = "model_weights/training.json"
        self.num_states = num_states
        self.should_load_models = should_load_models
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.batch_size=batch_size
        self.num_hidden_layers = num_hidden_layers
        self.lose_rate = 1e-4
        self.epsilon_clip = ep_clip
        global epsilon_clip
        epsilon_clip = ep_clip
        self.distribution = NormalDistribution(num_actions=num_actions)
        self.use_conv = use_conv
        self.build_actor_and_critic()
        self.gamma = gamma
        self.lam = lam
        self.entropy_coeff = entropy_coeff
        self.epochs = epochs
        self.train_lock = threading.Lock()
        self.dummy_action=np.zeros((1,self.num_actions))
        self.dummy_value=np.zeros((1, 1))
        global var
        var = 0.6
        
    def build_actor_and_critic(self):
        self.build_actor()
        self.build_critic()        
        if self.should_load_models:
            self.load_model_weights()
        else:
            self.training_info = {"episode" : 0}
        self.optimizer = tf.keras.optimizers.Adam()

    def build_actor(self):
        inputs = tf.keras.Input(shape=(self.num_states,))
        advantage = tf.keras.Input(shape=(1,))
        old_pred = tf.keras.Input(shape=(self.num_actions,))

        x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(inputs)
        for _ in range(self.num_hidden_layers - 1):
            x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(x)
        out_actor = tf.keras.layers.Dense(self.num_actions, kernel_initializer=tf.random_normal_initializer())(x)
        self.actor = tf.keras.models.Model(inputs=[inputs, advantage, old_pred], outputs=[out_actor])
        self.actor.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=[ppo_loss_continuous(
                    advantage=advantage,
                    old_pred=old_pred)],
                experimental_run_tf_function=False)
        self.actor.summary()

    def build_critic(self):
        inputs = tf.keras.Input(shape=(self.num_states,))
        x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(inputs)
        for _ in range(self.num_hidden_layers - 1):
            x = tf.keras.layers.Dense(self.hidden_size, activation="relu")(x)
        out_critic = tf.keras.layers.Dense(1, kernel_initializer=tf.random_normal_initializer())(x)
        self.critic = tf.keras.models.Model(inputs=[inputs], outputs=[out_critic])
        self.critic.compile(optimizer=tf.keras.optimizers.Adam(), loss="mse")
        self.critic.summary()

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
        return self.distribution.sample(), self.critic([observ, self.dummy_action, self.dummy_value])
    

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
            
            observ_arr = np.array(ep_dic["observations"])
            observ_arr = np.reshape(observ_arr, (observ_arr.shape[0], observ_arr.shape[2]))
            #print("OBSERVATIONS: ", observ_arr)
            ep_dic["adv"] = np.reshape(ep_dic["adv"], (ep_dic["adv"].shape[0], 1))
            #print("ADVANTAGE: ", ep_dic["adv"])
            ep_dic["means"] = np.array([mean.numpy()[0] for mean in ep_dic["means"]])
            ep_dic["actions"] = np.array([action.numpy()[0] for action in ep_dic["actions"]])
            #print("OLD PREDICTIONS: ", ep_dic["means"])
            #print("ACTION: ", ep_dic["actions"])
            #print("REWARD: ", ep_dic["tdlamret"])

            self.actor.fit([observ_arr, ep_dic["adv"], ep_dic["means"]], ep_dic["actions"], batch_size=self.batch_size, epochs=self.epochs)
            self.critic.fit(observ_arr, ep_dic["tdlamret"], batch_size=self.batch_size, epochs=self.epochs)
            self.training_info["episode"] += 1
            self.save_model_weights()
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

    def save_model_weights(self):
        self.actor.save_weights("model_weights/actor_model")
        self.critic.save_weights("model_weights/critic_model")
        with open(self.training_json, "w") as f:
            json.dump(self.training_info, f)


    def load_model_weights(self):
        self.actor.load_weights("model_weights/actor_model")
        self.critic.load_weights("model_weights/critic_model")
        self.optimizer = tf.keras.optimizers.Adam()
        
        with open(self.training_json, "r") as f:
            self.training_info = json.load(f)

    # def update_old_model(self):
    #     self.actor_old = tf.keras.models.clone_model(self.actor)
    #     self.actor_old.set_weights(self.actor.get_weights())

#ppo_model = PPOModel(10)
#ppo_model.build_actor_and_critic()
