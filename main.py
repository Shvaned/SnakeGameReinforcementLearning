"""Training loop for DQN Snake.
Saves the best model (highest score) to --save-dir/best_model.pth and also logs progress to CSV.
"""
import argparse
import os
import time
import numpy as np
from game import SnakeGame
from agent import Agent
from helper import plot, append_log

def train(episodes, save_dir, target_update=10, log_csv='logs/train_log.csv'):
    env = SnakeGame()
    agent = Agent()

    best_score = 0
    scores = []
    mean_scores = []
    total_score = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_score = 0
        losses = []

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, score = env.play_step(action)
            agent.remember(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            state = next_state
            ep_score = score

        agent.n_games += 1
        if ep % target_update == 0:
            agent.update_target()

        scores.append(ep_score)
        total_score += ep_score
        mean_scores.append(total_score / len(scores))

        # logging
        avg_loss = float(np.mean(losses)) if losses else None
        row = {
            'episode': ep,
            'score': ep_score,
            'mean_score': mean_scores[-1],
            'best_score': best_score,
            'avg_loss': avg_loss,
            'epsilon': agent.epsilon
        }
        append_log(log_csv, row)

        if ep_score > best_score:
            best_score = ep_score
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, 'best_model.pth')
            agent.save(model_path, extra={'episode': ep, 'score': ep_score})

        if ep % 10 == 0:
            print(f"Episode {ep} | Score {ep_score} | Mean {mean_scores[-1]:.2f} | Best {best_score}")
            plot(scores, mean_scores)

    print('Training complete')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=500, help='number of episodes to train')
    parser.add_argument('--save-dir', type=str, default='saved', help='where to save best model')
    parser.add_argument('--target-update', type=int, default=10, help='target network update frequency (episodes)')
    args = parser.parse_args()
    train(args.episodes, args.save_dir, args.target_update)
