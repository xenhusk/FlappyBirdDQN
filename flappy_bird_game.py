import pygame
import random
import sys

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE = (135, 206, 235)
YELLOW = (255, 255, 0)
DARK_GREEN = (0, 128, 0)
BROWN = (139, 69, 19)
RED = (255, 0, 0)

class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 20
        self.velocity = 0
        self.gravity = 0.8
        self.jump_strength = -12
        self.color = YELLOW
        
    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity
        
    def jump(self):
        self.velocity = self.jump_strength
        
    def draw(self, screen):
        # Draw bird body with gradient effect
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, BLACK, (int(self.x), int(self.y)), self.radius, 2)
        
        # Draw inner body for depth
        inner_radius = self.radius - 3
        pygame.draw.circle(screen, (255, 255, 100), (int(self.x), int(self.y)), inner_radius)
        
        # Draw eye (facing right - direction of flight) - positioned on the front of the bird
        eye_x = int(self.x + 8)  # Move to the right side (front of bird)
        eye_y = int(self.y - 5)
        pygame.draw.circle(screen, WHITE, (eye_x, eye_y), 5)
        pygame.draw.circle(screen, BLACK, (eye_x + 1, eye_y), 2)  # Pupil slightly to the right
        
        # Draw beak (pointing right - direction of flight) - positioned at the front
        beak_x = int(self.x + self.radius)  # Start at the edge of the bird's body
        beak_y = int(self.y)
        beak_points = [(beak_x, beak_y - 4),      # Top of beak base
                    (beak_x + 12, beak_y),      # Tip of beak (extends to the right)
                    (beak_x, beak_y + 4)]       # Bottom of beak base
        pygame.draw.polygon(screen, RED, beak_points)
        
        # Draw wing for more detail - positioned on the back/left side
        wing_x = int(self.x - 5)  # Move to left side (back of bird)
        wing_y = int(self.y + 2)
        pygame.draw.ellipse(screen, (200, 200, 0), (wing_x - 6, wing_y - 3, 12, 6))
        
        # Draw tail feathers for better direction indication - at the back (left side)
        tail_x = int(self.x - self.radius + 2)  # Position at the back (left side)
        tail_y = int(self.y)
        pygame.draw.ellipse(screen, (180, 180, 0), (tail_x - 8, tail_y - 2, 8, 4))  # Extend to the left
        
    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius, 
                          self.radius * 2, self.radius * 2)

class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = 60
        self.gap = 150
        self.height = random.randint(100, SCREEN_HEIGHT - self.gap - 100)
        self.speed = 3
        self.passed = False
        
    def update(self):
        self.x -= self.speed
        
    def draw(self, screen):
        # Upper pipe with gradient effect
        pygame.draw.rect(screen, DARK_GREEN, 
                        (self.x, 0, self.width, self.height))
        pygame.draw.rect(screen, BLACK, 
                        (self.x, 0, self.width, self.height), 2)
        
        # Upper pipe cap with 3D effect
        pygame.draw.rect(screen, DARK_GREEN, 
                        (self.x - 5, self.height - 25, self.width + 10, 25))
        pygame.draw.rect(screen, BLACK, 
                        (self.x - 5, self.height - 25, self.width + 10, 25), 2)
        
        # Upper cap highlight
        pygame.draw.rect(screen, (0, 160, 0), 
                        (self.x - 4, self.height - 24, self.width + 8, 3))
        
        # Lower pipe with gradient effect
        lower_pipe_y = self.height + self.gap
        pygame.draw.rect(screen, DARK_GREEN, 
                        (self.x, lower_pipe_y, self.width, SCREEN_HEIGHT - lower_pipe_y))
        pygame.draw.rect(screen, BLACK, 
                        (self.x, lower_pipe_y, self.width, SCREEN_HEIGHT - lower_pipe_y), 2)
        
        # Lower pipe cap with 3D effect
        pygame.draw.rect(screen, DARK_GREEN, 
                        (self.x - 5, lower_pipe_y, self.width + 10, 25))
        pygame.draw.rect(screen, BLACK, 
                        (self.x - 5, lower_pipe_y, self.width + 10, 25), 2)
        
        # Lower cap highlight
        pygame.draw.rect(screen, (0, 160, 0), 
                        (self.x - 4, lower_pipe_y + 1, self.width + 8, 3))
    
    def get_rects(self):
        upper_rect = pygame.Rect(self.x, 0, self.width, self.height)
        lower_rect = pygame.Rect(self.x, self.height + self.gap, 
                                self.width, SCREEN_HEIGHT - self.height - self.gap)
        return upper_rect, lower_rect
    
    def is_off_screen(self):
        return self.x + self.width < 0

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.big_font = pygame.font.Font(None, 72)
        self.reset_game()
        
    def reset_game(self):
        self.bird = Bird(100, SCREEN_HEIGHT // 2)
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.game_started = False
        if not hasattr(self, 'high_score'):
            self.high_score = 0
        
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.game_started:
                        self.game_started = True
                    elif self.game_over:
                        self.reset_game()
                    else:
                        self.bird.jump()
                elif event.key == pygame.K_ESCAPE:
                    return False
        return True
    
    def update(self):
        if not self.game_started or self.game_over:
            return
            
        # Update bird
        self.bird.update()
        
        # Check if bird hits ground or ceiling
        if (self.bird.y + self.bird.radius >= SCREEN_HEIGHT or 
            self.bird.y - self.bird.radius <= 0):
            self.game_over = True
            
        # Generate pipes
        if len(self.pipes) == 0 or self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(Pipe(SCREEN_WIDTH))
            
        # Update pipes
        for pipe in self.pipes[:]:
            pipe.update()
            
            # Check collision
            bird_rect = self.bird.get_rect()
            upper_rect, lower_rect = pipe.get_rects()
            
            if bird_rect.colliderect(upper_rect) or bird_rect.colliderect(lower_rect):
                self.game_over = True
                
            # Check if bird passed pipe
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                # Update high score
                if self.score > self.high_score:
                    self.high_score = self.score
                
            # Remove off-screen pipes
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
    
    def draw_background(self):
        # Enhanced sky gradient
        for y in range(SCREEN_HEIGHT):
            color_ratio = y / SCREEN_HEIGHT
            r = int(135 + (100 - 135) * color_ratio)
            g = int(206 + (150 - 206) * color_ratio)
            b = int(235 + (200 - 235) * color_ratio)
            pygame.draw.line(self.screen, (r, g, b), (0, y), (SCREEN_WIDTH, y))
            
        # Draw enhanced clouds
        for i in range(4):
            x = (i * 120 + 30) % SCREEN_WIDTH
            y = 40 + i * 25
            # Main cloud body
            pygame.draw.ellipse(self.screen, WHITE, (x, y, 90, 45))
            pygame.draw.ellipse(self.screen, WHITE, (x + 15, y - 8, 70, 40))
            pygame.draw.ellipse(self.screen, WHITE, (x + 30, y - 15, 50, 35))
            # Cloud shadow
            pygame.draw.ellipse(self.screen, (200, 200, 200), (x + 2, y + 2, 90, 45))
            
        # Draw ground
        ground_y = SCREEN_HEIGHT - 20
        pygame.draw.rect(self.screen, (139, 69, 19), (0, ground_y, SCREEN_WIDTH, 20))
        pygame.draw.rect(self.screen, (160, 82, 45), (0, ground_y, SCREEN_WIDTH, 5))
    
    def draw(self):
        self.draw_background()
        
        # Draw pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)
            
        # Draw bird
        self.bird.draw(self.screen)
        
        # Draw enhanced score display
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_text.get_rect()
        score_rect.topleft = (10, 10)
        
        # Score background with border
        pygame.draw.rect(self.screen, (0, 0, 0, 180), score_rect.inflate(20, 10))
        pygame.draw.rect(self.screen, WHITE, score_rect.inflate(20, 10), 2)
        self.screen.blit(score_text, score_rect)
        
        # Draw high score if available
        if hasattr(self, 'high_score') and self.high_score > 0:
            high_score_text = self.font.render(f"Best: {self.high_score}", True, YELLOW)
            high_score_rect = high_score_text.get_rect()
            high_score_rect.topleft = (10, 50)
            pygame.draw.rect(self.screen, (0, 0, 0, 180), high_score_rect.inflate(20, 10))
            pygame.draw.rect(self.screen, YELLOW, high_score_rect.inflate(20, 10), 2)
            self.screen.blit(high_score_text, high_score_rect)
        
        # Draw enhanced start screen
        if not self.game_started:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(150)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            # Title with shadow effect
            title = self.big_font.render("FLAPPY BIRD", True, YELLOW)
            title_rect = title.get_rect(center=(SCREEN_WIDTH//2 + 3, SCREEN_HEIGHT//2 - 50 + 3))
            title_shadow = self.big_font.render("FLAPPY BIRD", True, BLACK)
            self.screen.blit(title_shadow, title_rect)
            title_rect = title.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
            self.screen.blit(title, title_rect)
            
            # Subtitle
            subtitle = self.font.render("AI-Powered DQN Agent", True, WHITE)
            subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 10))
            self.screen.blit(subtitle, subtitle_rect)
            
            instruction = self.font.render("Press SPACE to start", True, WHITE)
            instruction_rect = instruction.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 30))
            self.screen.blit(instruction, instruction_rect)
            
            controls = self.font.render("SPACE to flap, ESC to quit", True, WHITE)
            controls_rect = controls.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 70))
            self.screen.blit(controls, controls_rect)
        
        # Draw enhanced game over screen
        elif self.game_over:
            overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            overlay.set_alpha(150)
            overlay.fill(BLACK)
            self.screen.blit(overlay, (0, 0))
            
            # Game over text with shadow
            game_over_text = self.big_font.render("GAME OVER", True, RED)
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2 + 3, SCREEN_HEIGHT//2 - 80 + 3))
            game_over_shadow = self.big_font.render("GAME OVER", True, BLACK)
            self.screen.blit(game_over_shadow, game_over_rect)
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 80))
            self.screen.blit(game_over_text, game_over_rect)
            
            # Score display
            final_score = self.font.render(f"Final Score: {self.score}", True, WHITE)
            score_rect = final_score.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 30))
            self.screen.blit(final_score, score_rect)
            
            # High score if available
            if hasattr(self, 'high_score') and self.high_score > 0:
                high_score_text = self.font.render(f"Best Score: {self.high_score}", True, YELLOW)
                high_score_rect = high_score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 10))
                self.screen.blit(high_score_text, high_score_rect)
            
            restart = self.font.render("Press SPACE to restart", True, WHITE)
            restart_rect = restart.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))
            self.screen.blit(restart, restart_rect)
            
            quit_text = self.font.render("Press ESC to quit", True, WHITE)
            quit_rect = quit_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 80))
            self.screen.blit(quit_text, quit_rect)
    
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            pygame.display.flip()
            self.clock.tick(FPS)
            
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    # Check if pygame is installed
    try:
        import pygame
    except ImportError:
        print("Pygame is not installed. Please install it using:")
        print("pip install pygame")
        sys.exit(1)
    
    game = Game()
    game.run()