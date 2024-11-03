import pygame
import numpy as np
import random

# Constants
SCREEN_WIDTH = 520
SCREEN_HEIGHT = 768
CLOCK = pygame.time.Clock()
CELL_SIZE = 64
FPS = 60
BLACK = (0, 0, 0)
RED = (255, 0, 0)
SCROLL_SPEED = 3
GRAVITY = .1

## PYGAME INITIALIZING
pygame.init()
WIN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Placeholderhere's Flappy Bird")

## CLASSES
class Pipes:
    def __init__(self, x, y1, y2, sprite1, sprite2):
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.sprite1 = sprite1
        self.size1 = sprite1.get_size()
        self.sprite2 = sprite2
        self.size2 = sprite2.get_size()
        self.score_point = False

    def blit(self, win):
        win.blit(self.sprite1, (self.x, self.y1))
        win.blit(self.sprite2, (self.x, self.y2))

    def collision_detection(self, colliding_rect):
        pipe_rect = pygame.Rect(self.x, 0, self.size1[0], SCREEN_HEIGHT)
        top_pipe_rect = pygame.Rect(self.x, self.y1, self.size1[0], self.size1[1])
        bottom_pipe_rect = pygame.Rect(self.x, self.y2, self.size2[0], self.size2[1])

        if pipe_rect.colliderect(colliding_rect):
            if top_pipe_rect.colliderect(colliding_rect):
                return "HIT"
            elif bottom_pipe_rect.colliderect(colliding_rect):
                return "HIT"
            else:
                return "SCORE"
        else:
            return False
class Bird:
    def __init__(self, x, y, sprite):
        self.x = x
        self.y = y
        self.size = sprite.get_size()
        self.sprite = sprite
        self.velocityY = 0
        self.alive = True

    def blit(self, win):
        win.blit(self.sprite, (self.x, self.y))

    def flap(self):
        self.velocityY -= 4

## FUNCTIONS
def display_text(win, x, y, msg, color):
    pygame.font.init()

    font = pygame.font.SysFont('Arial', 35)
    text = font.render(msg, False, color)
    text_rect = text.get_rect(center=(x, y))

    win.blit(text, text_rect)

def generate_Pipe(base_height):
    pipe_gap = random.randint(150, 180)
    sizeY1 = random.randint(128, SCREEN_HEIGHT - pipe_gap - base_height - 128)
    return sizeY1, pipe_gap

def generate_weights(seed, starting_weights, starting_bias, num_input, num_neurons, num_output):
    if seed != False:
        np.random.seed(seed)
        empty_weights = [np.zeros((input_size, num_neurons)), np.zeros((num_neurons, output_size))]
    # [0] = input -> hidden; [1] = hidden -> output
    random_weight_adjustment = [np.random.randn(num_input, num_neurons), np.random.randn(num_neurons, num_output)]
    random_bias_adjustment = [np.random.randn(num_neurons)]
    weights = [starting_weights[0] + random_weight_adjustment[0], starting_weights[1] + random_weight_adjustment[1]]
    bias = starting_bias + random_bias_adjustment
    return [bias, weights]

def ReLU(x):
    x[x < 0] = 0
    return x

def scale_to_range(matrix, new_min=-1, new_max=1):
    min_val = np.min(matrix)
    max_val = np.max(matrix)

    # Scale the matrix to the range [0, 1]
    scaled = (matrix - min_val) / (max_val - min_val)

    # Transform to the desired range [-1, 1]
    scaled = scaled * (new_max - new_min) + new_min
    return scaled

def calculate_output(input_matrix, weights, bias):
    # Activation of input layer
    a1 = input_matrix

    # Transition from Input -> Hidden Layer; z1 = Pre Activation
    z1 = np.matmul(a1, weights[0]) + bias

    # Activation of Hidden Layer
    a2 = ReLU(z1)

    # Transition from layer 2 (hidden layer) -> layer 3 (output layer)
    z2 = np.matmul(a2, weights[1])

    # s = e^z2[i] Output Layer
    a3 = ReLU(z2)
    column_average = np.sum(a3, axis=0)
    return column_average

## GAME VARIABLES
running = True
score = 0
high_score = 0
num_iterations = 0
run_game = False

base_height = 128
base1X = 0
base2X = SCREEN_WIDTH
baseY = SCREEN_HEIGHT - 128

## SPRITES
base_img = pygame.transform.scale(pygame.image.load('base.png'), (SCREEN_WIDTH, base_height))
background_img = pygame.transform.scale(pygame.image.load('bg.png'), (SCREEN_WIDTH, SCREEN_HEIGHT))
bottom_pipe_img = pygame.transform.scale(pygame.image.load('pipe.png'), (78, SCREEN_HEIGHT))
top_pipe_img = pygame.transform.rotate(bottom_pipe_img, 180)
bird_img1 = pygame.transform.scale(pygame.image.load('bird1.png'), (68, 48))

## CLASSES
pipe_spacing = 400
pipes = [Pipes(SCREEN_WIDTH, 0, 0, top_pipe_img, bottom_pipe_img), Pipes(SCREEN_WIDTH + pipe_spacing, 0, 0, top_pipe_img, bottom_pipe_img), Pipes(SCREEN_WIDTH + (2 * pipe_spacing), 0, 0, top_pipe_img, bottom_pipe_img)]
for pipe_index, pipe in enumerate(pipes):
    sizeY1, gap = generate_Pipe(base_height)
    pipe.y1 = -(SCREEN_HEIGHT - sizeY1)
    pipe.y2 = pipe.y1 + SCREEN_HEIGHT + gap

## """MACHINE LEARNING"""
# Input: playerX, sizeX, playerY, sizeY, (pipeX, pipe_sizeX, pipe_gapY, gap_sizeY) * 3 pipes
num_neurons = 8
input_size = 16
output_size = 2
num_bots = 64
dead_bots = 0

## Generate Random Weights
bot_variables = []
empty_bias = np.zeros((num_neurons, 1))
empty_weights = [np.zeros((input_size, num_neurons)), np.zeros((num_neurons, output_size))]
for i in range(num_bots):
    # Score, Bird(Class), [bias, [W0, W1]]
    bot_variables.append([0, Bird(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2, bird_img1), generate_weights(seed=False,
        starting_weights=empty_weights, starting_bias=empty_bias, num_input=input_size, num_neurons=num_neurons,
        num_output=output_size)])

# ## Load Previous Weights
# num_iterations = np.load('iterations.npy')
# pygame.display.set_caption(f"Placeholderhere's Flappy Bird; Iterations: {num_iterations}")
# bot_variables = [[0, Bird(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2, bird_img1), [np.load('bias.npy'), [np.load('W0.npy'), np.load('W1.npy')]]],]
# for i in range(num_bots - 1):
#     bot_variables.append(
#         [0, Bird(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2, bird_img1), generate_weights(seed=False,
#             starting_weights=[np.load('W0.npy'), np.load('W1.npy')],
#             starting_bias=np.load('bias.npy'),
#             num_input=input_size,
#             num_neurons=num_neurons,
#             num_output=output_size)])

## GAME LOOP
while running:
    CLOCK.tick(FPS)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                run_game = not run_game

    # WIN.fill(BLACK)
    if run_game:
        ## Background
        WIN.blit(background_img, (0, 0))
        score += 1

        ## PIPES
        for pipe_index, pipe in enumerate(pipes):
            pipe.x -= SCROLL_SPEED
            if pipe.x < -100:
                sizeY1, gap = generate_Pipe(base_height)
                pipe.x = pipes[pipe_index - 1].x + pipe_spacing
                pipe.y1 = -(SCREEN_HEIGHT - sizeY1)
                pipe.y2 = pipe.y1 + SCREEN_HEIGHT + gap

            # Collision with bots
            for bot in bot_variables:
                if bot[1].alive:
                    pipe_collsion = pipe.collision_detection(
                        pygame.Rect(bot[1].x + 8, bot[1].y + 8, bot[1].size[0], bot[1].size[1]))

                    if pipe_collsion == "HIT":
                        bot[0] = score
                        dead_bots += 1
                        bot[1].alive = False

            pipe.blit(WIN)

        ## BOTS
        # Bot = [score, Bird(Class), weights]
        if dead_bots < num_bots:
            for bot_index, bot in enumerate(bot_variables):
                if bot[1].alive:
                    if bot[1].y + bot[1].size[1] > baseY:
                        bot[0] = score
                        bot[1].alive = False
                        dead_bots += 1
                        break

                    if bot[1].y < 0:
                        bot[0] = score
                        bot[1].alive = False
                        dead_bots += 1
                        break

                    bot[1].velocityY += GRAVITY
                    bot[1].y += bot[1].velocityY
                    input_matrix = [bot[1].x, bot[1].size[0], bot[1].y, bot[1].size[1],
                                    pipes[0].x, pipes[0].y1 + SCREEN_HEIGHT, pipes[0].y2 - (pipes[0].y1 + SCREEN_HEIGHT),
                                    pipes[0].size1[0],
                                    pipes[1].x, pipes[1].y1 + SCREEN_HEIGHT, pipes[1].y2 - (pipes[1].y1 + SCREEN_HEIGHT),
                                    pipes[1].size1[0],
                                    pipes[2].x, pipes[2].y1 + SCREEN_HEIGHT, pipes[2].y2 - (pipes[2].y1 + SCREEN_HEIGHT),
                                    pipes[2].size1[0]]

                    scaled_input_matrix = scale_to_range(input_matrix, -1, 1)
                    # Input, Weights, Bias
                    output_matrix = calculate_output(scaled_input_matrix, bot[2][1], bot[2][0])

                    if output_matrix[1] <= output_matrix[0]:
                        bot[1].flap()

                    bot[1].blit(WIN)

        elif dead_bots >= num_bots:
            bot_variables.sort(key=lambda x: x[0])
            if high_score < bot_variables[-1][0]:
                high_score = bot_variables[-1][0]
            best_weights = bot_variables[-1][2]
            W0, W1 = best_weights[1]
            b = best_weights[0]
            np.save('iterations.npy', num_iterations)
            np.save('W0.npy', W0)
            np.save('W1.npy', W1)
            np.save('bias.npy', b)


            ## RESET VARIABLES
            # SCORE
            score = 0

            # PIPES
            pipes = [Pipes(SCREEN_WIDTH, 0, 0, top_pipe_img, bottom_pipe_img),
                     Pipes(SCREEN_WIDTH + pipe_spacing, 0, 0, top_pipe_img, bottom_pipe_img),
                     Pipes(SCREEN_WIDTH + (2 * pipe_spacing), 0, 0, top_pipe_img, bottom_pipe_img)]
            for pipe_index, pipe in enumerate(pipes):
                sizeY1, gap = generate_Pipe(base_height)
                pipe.y1 = -(SCREEN_HEIGHT - sizeY1)
                pipe.y2 = pipe.y1 + SCREEN_HEIGHT + gap

            # BOT VARIABLES
            num_iterations += 1
            dead_bots = 0
            bot_variables = [[0, Bird(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2, bird_img1), best_weights]]
            for i in range(num_bots - 1):
                bot_variables.append(
                    [0, Bird(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2, bird_img1), generate_weights(seed=False,
                        starting_weights=[W0, W1],
                        starting_bias=b,
                        num_input=input_size,
                        num_neurons=num_neurons,
                        num_output=output_size)])

            pygame.display.set_caption(f"Placeholderhere's Flappy Bird; Iteration: {num_iterations}")

        ## BASE
        base1X -= SCROLL_SPEED
        base2X -= SCROLL_SPEED
        if base1X < -SCREEN_WIDTH:
            base1X = SCREEN_WIDTH - SCROLL_SPEED
        if base2X < -SCREEN_WIDTH:
            base2X = SCREEN_WIDTH - SCROLL_SPEED

        WIN.blit(base_img, (base1X, SCREEN_HEIGHT - base_height))
        WIN.blit(base_img, (base2X, SCREEN_HEIGHT - base_height))

        ## SCORE
        display_text(WIN, SCREEN_WIDTH / 2, SCREEN_HEIGHT - (base_height // 2), f'Score: {score}', BLACK)
        display_text(WIN, SCREEN_WIDTH / 2, SCREEN_HEIGHT - (base_height // 2) + 32, f'high_score: {high_score}', BLACK)

    else:
        display_text(WIN, SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2, 'PAUSED', RED)

    ## Update Display
    pygame.display.update()