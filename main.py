import gym
import random
import numpy as np
import os
import json

def save_state(qtable, epsilon, learning_rate, discount_rate, decay_rate, filename="state.json"):
    with open(filename, 'w') as f:
        json.dump({
            "qtable": qtable.tolist(),
            "epsilon": epsilon,
            "learning_rate": learning_rate,
            "discount_rate": discount_rate,
            "decay_rate": decay_rate
        }, f)

def load_state(filename="state.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
            qtable = np.array(data["qtable"])
            epsilon = data["epsilon"]
            learning_rate = data["learning_rate"]
            discount_rate = data["discount_rate"]
            decay_rate = data.get("decay_rate")
            return qtable, epsilon, learning_rate, discount_rate, decay_rate, True
    return None, None, None, None, None, False

def main():
    env = gym.make("Taxi-v3", render_mode="human")
    state_size = env.observation_space.n
    action_size = env.action_space.n

    qtable, epsilon, learning_rate, discount_rate, decay_rate, state_loaded = load_state()
    if not state_loaded:
        qtable = np.zeros((state_size, action_size))
        epsilon = 0.05
        learning_rate = 0.9
        discount_rate = 0.8
        decay_rate = 0.005
    else:
        print("The model loaded correctly")
        print(epsilon)

    EPISODES = 1000
    STEPS_PER_EPISODE = 99

    for episode in range(EPISODES):
        done = False
        state = env.reset()[0]

        for step in range(STEPS_PER_EPISODE):
            if random.uniform(0, 1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state, :])

            # do the action
            new_state, reward, done, info, *_ = env.step(action)
            # update the qtable
            qtable[state, action] = qtable[state, action] + learning_rate * (
                        reward + discount_rate * np.max(qtable[new_state, :]) - qtable[state, action])
            state = new_state

            if done:
                break

        #epsilon = np.exp(-decay_rate * episode)
        save_state(qtable, epsilon, learning_rate, discount_rate, decay_rate)

    env.close()

if __name__ == "__main__":
    main()
