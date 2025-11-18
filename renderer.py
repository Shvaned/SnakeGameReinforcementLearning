"""Simple pygame renderer to watch the agent play.
Usage:
    python renderer.py --model saved/best_model.pth --episodes 5
If --model is omitted, you can control the snake with arrow keys.
"""
import argparse
import pygame
import sys
import time
from game import SnakeGame
from model import load_model
import torch

SCALE = 1
BG_COLOR = (0,0,0)
SNAKE_COLOR = (0,255,0)
FOOD_COLOR = (255,0,0)
GRID_COLOR = (40,40,40)

def draw_game(screen, env):
    block = env.BLOCK
    w, h = env.w, env.h
    screen.fill(BG_COLOR)
    # draw grid (optional)
    for x in range(0, w, block):
        pygame.draw.line(screen, GRID_COLOR, (x,0), (x,h))
    for y in range(0, h, block):
        pygame.draw.line(screen, GRID_COLOR, (0,y), (w,y))

    # draw food
    fx, fy = env.food
    pygame.draw.rect(screen, FOOD_COLOR, pygame.Rect(fx, fy, block, block))

    # draw snake
    for idx, p in enumerate(env.snake):
        rect = pygame.Rect(p.x, p.y, block, block)
        if idx == 0:
            pygame.draw.rect(screen, (0,200,0), rect)
        else:
            pygame.draw.rect(screen, SNAKE_COLOR, rect)

    # draw score
    font = pygame.font.SysFont(None, 24)
    img = font.render(f'Score: {env.score}', True, (200,200,200))
    screen.blit(img, (10,10))

    pygame.display.flip()

def human_action_from_keys(keys):
    # returns action one-hot based on arrow keys relative to current direction
    # We'll map keys to absolute direction changes: left, right, up, down -> convert to relative
    move = None
    if keys[pygame.K_LEFT]: move = 2  # left turn
    if keys[pygame.K_RIGHT]: move = 1  # right turn
    if keys[pygame.K_UP] or keys[pygame.K_DOWN]:
        # up/down mapped to straight for simplicity (user expects absolute control)
        move = 0
    if move is None:
        return [1,0,0]
    arr = [0,0,0]
    arr[move] = 1
    return arr

def run(model_path=None, episodes=1, fps=10):
    pygame.init()
    env = SnakeGame()
    screen = pygame.display.set_mode((env.w, env.h))
    clock = pygame.time.Clock()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = None
    if model_path:
        model = load_model(model_path, device)
        model.eval()

    for ep in range(episodes):
        state = env.reset()
        done = False
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

            if model is None:
                keys = pygame.key.get_pressed()
                action = human_action_from_keys(keys)
            else:
                s = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
                with torch.no_grad():
                    q = model(s)
                action_idx = int(torch.argmax(q).item())
                action = [0, 0, 0]
                action[action_idx] = 1

            # Step the environment
            _, reward, done, score = env.play_step(action)
            state = env.get_state()  # <-- Required for correct inference

            draw_game(screen, env)
            clock.tick(fps)

        # small pause between episodes
        time.sleep(0.5)
    pygame.quit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, help='path to model (optional)')
    parser.add_argument('--episodes', type=int, default=1, help='episodes to run')
    parser.add_argument('--fps', type=int, default=10, help='frames per second')
    args = parser.parse_args()
    run(args.model, args.episodes, args.fps)

#usage: python renderer.py --model saved/best_model.pth --episodes 5
