import gymnasium as gym
import tarware  # noqa: F401

def main():
    env_id = "tarware-tiny-3agvs-2pickers-partialobs-v1"
    env = gym.make(env_id)

    obs, info = env.reset(seed=21)

    print("env:", env_id)
    print("n_agents:", env.n_agents)
    print("action_space:", env.action_space)
    print("obs_len:", len(obs))

    for t in range(50):
        actions = env.action_space.sample()
        obs, reward, truncated, terminated, info = env.step(actions)

        if any(truncated) or any(terminated):
            print("episode finished at step", t)
            break

    env.close()
    print("TA-RWARE runs correctly")

if __name__ == "__main__":
    main()
