import gymnasium as gym
import neat
import pygame
import matplotlib.pyplot as plt
import numpy as np
FPS = 30
env = gym.make("MountainCar-v0", render_mode="rgb_array")
obs, info = env.reset(seed=42)

best_history = []
mean_history = []

config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    "config-pop-size-200.txt",
)

population = neat.Population(config)
population.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
population.add_reporter(stats)


def evaluate(genome, config, n_episodes=3):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    tot, successes = 0.0, 0
    for _ in range(n_episodes):
        env_ep = gym.make("MountainCar-v0")
        obs, _ = env_ep.reset(seed=None)
        ep = 0.0
        for _ in range(200):
            action = int(np.argmax(net.activate(obs)))
            obs, r, term, trunc, _ = env_ep.step(action)
            ep += r
            if term: successes += 1  # goal 
            if term or trunc: break
        env_ep.close()
        tot += ep
    genome.success_rate = successes / n_episodes  # optional
    return tot / n_episodes

def eval_genomes(genomes, config, record_video: bool = False):
    for _, genome in genomes:
        genome.fitness = evaluate(genome, config, n_episodes=3)

    best_f = max(g.fitness for _, g in genomes)
    mean_f = sum(g.fitness for _, g in genomes) / len(genomes)

    best_history.append(best_f)
    mean_history.append(mean_f)
    best_sr = max(getattr(g, "success_rate", 0.0) for _, g in genomes)
    print(f"Gen success rate (best): {best_sr:.2f}")



n_generations = 80
winner = population.run(eval_genomes, n_generations)

plt.figure()
plt.plot(range(len(best_history)), best_history, label="Best")
plt.plot(range(len(mean_history)), mean_history, label="Mean")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Fitness per Generation (MountainCar-v0)")
plt.legend()
plt.tight_layout()
plt.savefig("images/fitnesswith200popsize.png")
plt.show()

print("\nBest genome:\n{!s}".format(winner))

pygame.init()
screen = None
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 48, bold=True)
winner_text = font.render("WINNER!", True, (255, 215, 0))

def blit_frame(frame):
    global screen
    h, w, _ = frame.shape
    if screen is None:
        screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("MountainCar + WINNER overlay")
    surf = pygame.image.frombuffer(frame.tobytes(), (w, h), "RGB")
    screen.blit(surf, (0, 0))

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

obs, info = env.reset(seed=42)

for _ in range(300):
    action = int(np.argmax(winner_net.activate(obs)))
    obs, reward, terminated, truncated, info = env.step(action)

    frame = env.render()
    blit_frame(frame)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            env.close()
            raise SystemExit

    if terminated or truncated:
        rect = winner_text.get_rect(center=screen.get_rect().center)
        screen.blit(winner_text, rect)
        pygame.display.flip()
        pygame.time.wait(2000)
        obs, info = env.reset(seed=42) 
    pygame.display.flip()
    clock.tick(FPS)
env.close()
pygame.quit()