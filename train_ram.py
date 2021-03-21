import numpy as np
import gym
from agent import *
from config import *

def train(n_episodes, env, agent):
    rewards = []
    average_rewards = []
    
    for i in range(1, n_episodes+1):
        episodic_reward = 0
        state = env.reset()
        agent.noise.reset()
        
        action = agent.act(state, i)
        t=0
        
        while True:
            next_state, reward, done, _ = env.step(action)
            episodic_reward += reward
            agent.memory.add(state, action, reward, next_state, done)
            t += 1
            
            if len(agent.memory.memory)>agent.batch_size:
                if t % agent.step == 0:
                    for _ in range(agent.learning_time):
                        agent.learn()
            
            if done:
                break
                        
            state = next_state.copy()
            action = agent.act(state, i)
        
        rewards.append(episodic_reward)
        average_rewards.append(np.mean(rewards[-100:]))
        
        print('\rEpisode {}. Total score for this episode: {:.3f}, average score {:.3f}'.format(i, rewards[-1], average_rewards[-1]), end='')
        if i % 100 == 0:
            print('')
    
    return rewards, average_rewards

if __name__ == '__main__':
    env = gym.make(RAM_ENV_NAME)

    rewards = np.zeros((N, L))
    average_rewards = np.zeros((N, L))
    for i in range(N):
        print('{}/{}'.format(i+1, N))
        agent = DDPG_Agent(env=env, 
                           lr1=LR1, 
                           lr2=LR2, 
                           tau=TAU, 
                           speed1=SPEED1, 
                           speed2=SPEED2, 
                           step=STEP, 
                           learning_time=LEARNING_TIME,
                           batch_size=BATCH_SIZE, 
                           OUN_noise=OUN, 
                           batchnorm=BN, 
                           clip=CLIP,
                           initialize=INIT,
                           hidden=HIDDEN)
        rewards[i, :], average_rewards[i, :] = train(L, env, agent)
    torch.save(agent.actor_local.state_dict(), '{}_weights.pth'.format(RAM_ENV_NAME))
    np.save('./rewards_{}_{}_{}_{}_{}_{}.npy'.format(agent.OUN_noise, agent.batchnorm, agent.clip, agent.initialize, agent.hidden[0], agent.hidden[1]), rewards)
    np.save('./average_rewards_{}_{}_{}_{}_{}_{}.npy'.format(agent.OUN_noise, agent.batchnorm, agent.clip, agent.initialize, agent.hidden[0], agent.hidden[1]), average_rewards)