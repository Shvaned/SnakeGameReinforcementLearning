"""Evaluate a trained model over multiple episodes.
Reports: mean score, max score, std, distribution and prints a small CSV report.
"""
import argparse
import os
import numpy as np
from game import SnakeGame
from model import load_model
import torch

def evaluate(model_path, episodes=100, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = load_model(model_path, device)
    env = SnakeGame()

    scores = []
    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            s = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
            with torch.no_grad():
                q = model(s)
            action_idx = int(torch.argmax(q).item())
            action = [0,0,0]
            action[action_idx] = 1
            next_state, reward, done, score = env.play_step(action)
            state = next_state
        scores.append(score)

    scores = np.array(scores)
    print('Evaluation results')
    print(f'episodes: {episodes}')
    print(f'mean score: {scores.mean():.3f}')
    print(f'max score: {scores.max()}')
    print(f'min score: {scores.min()}')
    print(f'std: {scores.std():.3f}')
    # Save a small report
    report_path = os.path.splitext(model_path)[0] + '_eval.csv'
    import pandas as pd
    df = pd.DataFrame({'score': scores})
    df.to_csv(report_path, index=False)
    print(f'Report saved to {report_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='path to model (.pth)')
    parser.add_argument('--episodes', type=int, default=100, help='how many episodes to run')
    args = parser.parse_args()
    evaluate(args.model, args.episodes)
