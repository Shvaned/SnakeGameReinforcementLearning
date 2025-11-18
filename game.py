"""Headless Snake environment. Grid-based, deterministic moves.
State: 11 features (classic minimal representation used in many DQN Snake tutorials).
Action space: 3 actions -> [straight, right, left]
"""
from enum import Enum
from collections import namedtuple
import random

Point = namedtuple('Point', 'x y')

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGame:
    BLOCK = 20
    WIDTH = 640
    HEIGHT = 480

    def __init__(self, w=WIDTH, h=HEIGHT):
        self.w = w
        self.h = h
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        x = self.w // 2
        y = self.h // 2
        self.head = Point(x, y)
        self.snake = [self.head,
                      Point(x - self.BLOCK, y),
                      Point(x - 2 * self.BLOCK, y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        return self.get_state()

    def _place_food(self):
        max_x = (self.w - self.BLOCK) // self.BLOCK
        max_y = (self.h - self.BLOCK) // self.BLOCK
        self.food = Point(random.randint(0, max_x) * self.BLOCK,
                          random.randint(0, max_y) * self.BLOCK)
        if self.food in self.snake:
            self._place_food()

    def play_step(self, action):
        """action: one-hot like [1,0,0] -> straight, [0,1,0] -> right, [0,0,1] -> left"""
        self.frame_iteration += 1
        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        done = False

        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return self.get_state(), reward, done, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return self.get_state(), reward, done, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x < 0 or pt.x >= self.w or pt.y < 0 or pt.y >= self.h:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if action[0] == 1:
            new_dir = clock_wise[idx]
        elif action[1] == 1:
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir

        x, y = self.head
        if self.direction == Direction.RIGHT:
            x += self.BLOCK
        elif self.direction == Direction.LEFT:
            x -= self.BLOCK
        elif self.direction == Direction.UP:
            y -= self.BLOCK
        elif self.direction == Direction.DOWN:
            y += self.BLOCK

        self.head = Point(x, y)

    def get_state(self):
        head = self.head
        point_l = Point(head.x - self.BLOCK, head.y)
        point_r = Point(head.x + self.BLOCK, head.y)
        point_u = Point(head.x, head.y - self.BLOCK)
        point_d = Point(head.x, head.y + self.BLOCK)

        # Correct direction flags
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        # Danger checks
        if dir_r:
            danger_straight = self._is_collision(point_r)
            danger_right = self._is_collision(point_d)
            danger_left = self._is_collision(point_u)
        elif dir_l:
            danger_straight = self._is_collision(point_l)
            danger_right = self._is_collision(point_u)
            danger_left = self._is_collision(point_d)
        elif dir_u:
            danger_straight = self._is_collision(point_u)
            danger_right = self._is_collision(point_r)
            danger_left = self._is_collision(point_l)
        else:  # dir_d
            danger_straight = self._is_collision(point_d)
            danger_right = self._is_collision(point_l)
            danger_left = self._is_collision(point_r)

        food_left = self.food.x < head.x
        food_right = self.food.x > head.x
        food_up = self.food.y < head.y
        food_down = self.food.y > head.y

        state = [
            int(danger_straight),
            int(danger_right),
            int(danger_left),

            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),

            int(food_left),
            int(food_right),
            int(food_up),
            int(food_down)
        ]

        return state

