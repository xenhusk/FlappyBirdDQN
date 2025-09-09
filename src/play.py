import pygame
from envs.flappy_bird_env import FlappyBirdEnv

def play_flappy():
    env = FlappyBirdEnv()
    clock = pygame.time.Clock()
    
    # Show start screen
    font = pygame.font.Font(None, 36)
    
    # Initial reset
    state = env.reset()
    game_started = False
    done = False
    total_reward = 0
    
    print("Controls:")
    print("SPACE/UP ARROW - Start game and Flap")
    print("Q - Quit")
    print("R - Restart after game over")
    
    while True:
        if not game_started:
            # Draw start screen
            env.screen.fill(env.SKY_BLUE)
            start_text = font.render('Press SPACE to Start', True, env.BLACK)
            controls_text = font.render('SPACE/UP = Flap, Q = Quit', True, env.BLACK)
            env.screen.blit(start_text, (env.width//2 - start_text.get_width()//2, env.height//2))
            env.screen.blit(controls_text, (env.width//2 - controls_text.get_width()//2, env.height//2 + 50))
            pygame.display.flip()

        # Handle input
        action = 0  # Default action is do nothing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_SPACE, pygame.K_UP):
                    if not game_started:
                        game_started = True
                    action = 1  # Flap
                elif event.key == pygame.K_q:
                    pygame.quit()
                    return
                elif event.key == pygame.K_r and done:
                    # Reset the game
                    state = env.reset()
                    done = False
                    game_started = False
                    total_reward = 0
        
        if game_started and not done:
            # Step the environment
            state, reward, done, info = env.step(action)
            total_reward += reward
            
            if done:
                # Show game over screen
                game_over_text = font.render('Game Over! Press R to Restart', True, env.BLACK)
                score_text = font.render(f'Final Score: {info.get("score", 0)}', True, env.BLACK)
                env.screen.blit(game_over_text, (env.width//2 - game_over_text.get_width()//2, env.height//2))
                env.screen.blit(score_text, (env.width//2 - score_text.get_width()//2, env.height//2 + 50))
                pygame.display.flip()
        
        # Cap at 30 FPS for better control
        clock.tick(30)

if __name__ == "__main__":
    play_flappy()
