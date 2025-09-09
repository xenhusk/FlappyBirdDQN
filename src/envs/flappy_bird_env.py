import pygame
import numpy as np
import sys
import os
import math
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from flappy_bird_game import Game, Bird, Pipe, SCREEN_WIDTH, SCREEN_HEIGHT

class FlappyBirdEnvNew(Game):
    def __init__(self, training_mode=True, render_mode='none'):
        super().__init__()
        self.training_mode = training_mode
        self.render_mode = render_mode  # 'none', 'fast', 'human'
        self.training_fps = 1000 if training_mode else 60
        
        # Enhanced observation space (8 features instead of 4)
        self.observation_space_dim = 8
        self.action_space_dim = 2
        
        # Training statistics
        self.episode_count = 0
        self.total_steps = 0
        self.best_score = 0
        self.score_history = []
        
        # Curriculum learning parameters
        self.curriculum_phase = 0  # 0: Easy, 1: Medium, 2: Hard
        self.phase_thresholds = [500, 1500]  # Switch phases at these episode counts
        
        # Performance optimization
        self.headless = render_mode == 'none'
        if self.headless and training_mode:
            # For headless training, we don't need pygame display
            pygame.display.quit()
            
    def reset(self):
        """Reset environment for new episode"""
        super().reset_game()
        self.game_started = True
        self.episode_count += 1
        
        # Update curriculum phase
        self._update_curriculum_phase()
        
        # Track performance
        if hasattr(self, 'episode_score'):
            self.score_history.append(self.episode_score)
            if self.episode_score > self.best_score:
                self.best_score = self.episode_score
                
        self.episode_score = 0
        self.steps_in_episode = 0
        self.pipes_passed_in_episode = 0
        
        return self._get_enhanced_state()
    
    def _update_curriculum_phase(self):
        """Update curriculum learning phase based on episode count"""
        if self.episode_count <= self.phase_thresholds[0]:
            self.curriculum_phase = 0  # Easy phase
        elif self.episode_count <= self.phase_thresholds[1]:
            self.curriculum_phase = 1  # Medium phase
        else:
            self.curriculum_phase = 2  # Hard phase
    
    def _create_curriculum_pipe(self):
        """Create pipe based on current curriculum phase"""
        pipe = Pipe(SCREEN_WIDTH)
        
        if self.curriculum_phase == 0:  # Easy phase
            # Larger gaps, more centered
            pipe.gap = 200  # Larger gap
            pipe.height = random.randint(150, SCREEN_HEIGHT - pipe.gap - 150)
        elif self.curriculum_phase == 1:  # Medium phase
            # Normal gaps
            pipe.gap = 150
            pipe.height = random.randint(100, SCREEN_HEIGHT - pipe.gap - 100)
        else:  # Hard phase
            # Smaller gaps, more varied positions
            pipe.gap = 120
            pipe.height = random.randint(80, SCREEN_HEIGHT - pipe.gap - 80)
            
        return pipe
        
    def step(self, action):
        """Take action and return new state, reward, done, info"""
        # Handle action
        if action == 1:  # Flap action
            self.bird.jump()
            
        # Store previous state for reward calculation
        prev_score = self.score
        prev_bird_y = self.bird.y
        
        # Update game physics
        self._update_physics()
        
        # Calculate enhanced reward
        reward = self._calculate_enhanced_reward(prev_score, prev_bird_y)
        
        # Get new state
        state = self._get_enhanced_state()
        done = self.game_over
        
        # Update statistics
        self.total_steps += 1
        self.steps_in_episode += 1
        self.episode_score = self.score
        
        if self.score > prev_score:
            self.pipes_passed_in_episode += 1
        
        # Render based on mode
        self._render()
        
        # Create info dictionary with useful training information
        info = {
            'score': self.score,
            'episode': self.episode_count,
            'steps': self.steps_in_episode,
            'best_score': self.best_score,
            'bird_y': self.bird.y,
            'bird_velocity': self.bird.velocity,
            'pipes_passed': self.pipes_passed_in_episode
        }
        
        return state, reward, done, info
    
    def _update_physics(self):
        """Update game physics without rendering"""
        # Update bird
        self.bird.update()
        
        # Check boundaries
        if (self.bird.y + self.bird.radius >= SCREEN_HEIGHT or 
            self.bird.y - self.bird.radius <= 0):
            self.game_over = True
            return
            
        # Generate pipes with curriculum learning
        if len(self.pipes) == 0 or self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(self._create_curriculum_pipe())
            
        # Update pipes and check collisions
        for pipe in self.pipes[:]:
            pipe.update()
            
            # Collision detection
            if self._check_collision(pipe):
                self.game_over = True
                return
                
            # Score update
            if not pipe.passed and pipe.x + pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1
                
            # Remove off-screen pipes
            if pipe.is_off_screen():
                self.pipes.remove(pipe)
    
    def _check_collision(self, pipe):
        """More precise collision detection"""
        bird_rect = self.bird.get_rect()
        upper_rect, lower_rect = pipe.get_rects()
        return bird_rect.colliderect(upper_rect) or bird_rect.colliderect(lower_rect)
    
    def _calculate_enhanced_reward(self, prev_score, prev_bird_y):
        """Simplified and more effective reward function"""
        reward = 0.0
        
        # Base survival reward - small positive reward for staying alive
        reward += 0.1
        
        # Large reward for passing pipes
        if self.score > prev_score:
            reward += 10.0  # Strong positive reward for success
            return reward  # Early return for successful pipe passage
            
        # Penalty for death - but not too harsh initially
        if self.game_over:
            # Graduated penalty based on how long the bird survived
            survival_bonus = min(5.0, self.steps_in_episode * 0.01)  # Up to 5 bonus points
            reward = -1.0 + survival_bonus  # Start with small penalty, add survival bonus
            return reward
        
        # Get next pipe for positioning rewards
        next_pipe = self._get_next_pipe()
        if next_pipe:
            # Calculate bird's position relative to pipe gap
            gap_center = next_pipe.height + next_pipe.gap / 2
            gap_top = next_pipe.height
            gap_bottom = next_pipe.height + next_pipe.gap
            
            # Distance to pipe
            pipe_distance = next_pipe.x - self.bird.x
            
            # Only give positioning rewards when approaching the pipe
            if 0 <= pipe_distance <= 200:  # Within 200 pixels of pipe
                # Reward for being in the gap
                if gap_top < self.bird.y < gap_bottom:
                    # Closer to center is better
                    center_distance = abs(self.bird.y - gap_center)
                    max_distance = next_pipe.gap / 2
                    position_reward = 1.0 - (center_distance / max_distance)
                    reward += position_reward * 0.5
                else:
                    # Small penalty for being outside gap
                    reward -= 0.1
                    
                # Reward for maintaining good height when far from pipe
                if pipe_distance > 100:
                    ideal_height = SCREEN_HEIGHT * 0.5  # Center of screen
                    height_diff = abs(self.bird.y - ideal_height) / SCREEN_HEIGHT
                    if height_diff < 0.2:  # Within 20% of screen center
                        reward += 0.2
        
        # Small penalty for excessive velocity (encourage smooth movement)
        if abs(self.bird.velocity) > 10:
            reward -= 0.05
            
        # Store current action for next step
        self.last_action = getattr(self, 'last_action', 0)
            
        return reward
    
    def _get_next_pipe(self):
        """Get the next pipe the bird needs to navigate"""
        for pipe in self.pipes:
            if pipe.x + pipe.width >= self.bird.x - 10:  # Small buffer
                return pipe
        return None
    
    def _get_enhanced_state(self):
        """Enhanced state representation with more useful features"""
        # Get next pipe
        next_pipe = self._get_next_pipe()
        
        if next_pipe is None:
            # Default values when no pipe is present
            pipe_x = SCREEN_WIDTH
            pipe_gap_top = SCREEN_HEIGHT * 0.3
            pipe_gap_bottom = SCREEN_HEIGHT * 0.7
            pipe_gap_center = SCREEN_HEIGHT * 0.5
        else:
            pipe_x = next_pipe.x
            pipe_gap_top = next_pipe.height
            pipe_gap_bottom = next_pipe.height + next_pipe.gap
            pipe_gap_center = next_pipe.height + next_pipe.gap / 2
        
        # Enhanced state vector (8 features)
        state = np.array([
            # Bird position and dynamics
            self.bird.y / SCREEN_HEIGHT,  # Normalized bird height [0-1]
            self.bird.velocity / 15.0,    # Normalized velocity [-1 to 1]
            
            # Pipe information
            (pipe_x - self.bird.x) / SCREEN_WIDTH,  # Horizontal distance to pipe [0-1]
            pipe_gap_center / SCREEN_HEIGHT,        # Gap center position [0-1]
            
            # Gap boundaries
            pipe_gap_top / SCREEN_HEIGHT,           # Top of gap [0-1]
            pipe_gap_bottom / SCREEN_HEIGHT,        # Bottom of gap [0-1]
            
            # Relative positions
            (self.bird.y - pipe_gap_center) / SCREEN_HEIGHT,  # Bird position relative to gap center
            (pipe_x + next_pipe.width/2 - self.bird.x) / SCREEN_WIDTH if next_pipe else 1.0,  # Distance to pipe center
            
        ], dtype=np.float32)
        
        return state
    
    def _render(self):
        """Render based on current mode"""
        if self.render_mode == 'none':
            return
        elif self.render_mode == 'fast' and self.training_mode:
            # Render occasionally during training
            if self.steps_in_episode % 100 == 0 or self.game_over:
                if not self.headless:
                    self.draw()
                    pygame.display.flip()
        elif self.render_mode == 'human':
            # Full rendering for human viewing
            if not self.headless:
                self.draw()
                pygame.display.flip()
                self.clock.tick(60)
    
    def get_training_stats(self):
        """Get training statistics"""
        if len(self.score_history) == 0:
            return {
                'episodes': self.episode_count,
                'total_steps': self.total_steps,
                'best_score': self.best_score,
                'avg_score': 0.0,
                'recent_avg': 0.0
            }
            
        recent_scores = self.score_history[-100:] if len(self.score_history) >= 100 else self.score_history
        
        return {
            'episodes': self.episode_count,
            'total_steps': self.total_steps,
            'best_score': self.best_score,
            'avg_score': np.mean(self.score_history),
            'recent_avg': np.mean(recent_scores),
            'score_std': np.std(self.score_history),
            'recent_std': np.std(recent_scores)
        }
    
    def set_training_mode(self, training_mode=True):
        """Switch between training and evaluation mode"""
        self.training_mode = training_mode
        self.training_fps = 1000 if training_mode else 60
        
    def close(self):
        """Clean up pygame resources"""
        pygame.quit()

# Additional utility class for training management
class TrainingManager:
    """Utility class to manage training sessions"""
    
    def __init__(self, env):
        self.env = env
        self.training_log = []
        
    def log_episode(self, episode_reward, epsilon=None, loss=None):
        """Log episode results"""
        stats = self.env.get_training_stats()
        log_entry = {
            'episode': stats['episodes'],
            'reward': episode_reward,
            'score': stats['best_score'],
            'avg_score': stats['recent_avg'],
            'epsilon': epsilon,
            'loss': loss
        }
        self.training_log.append(log_entry)
        
    def print_progress(self, frequency=100):
        """Print training progress"""
        stats = self.env.get_training_stats()
        if stats['episodes'] % frequency == 0:
            print(f"Episode {stats['episodes']}: "
                  f"Best Score: {stats['best_score']}, "
                  f"Avg Score: {stats['recent_avg']:.2f} ± {stats.get('recent_std', 0):.2f}, "
                  f"Total Steps: {stats['total_steps']}")
    
    def should_save_model(self, threshold_score=5, frequency=1000):
        """Determine if model should be saved"""
        stats = self.env.get_training_stats()
        return (stats['best_score'] >= threshold_score and 
                stats['episodes'] % frequency == 0)

# Example usage function
def create_training_environment():
    """Create and configure training environment"""
    env = FlappyBirdEnvNew(
        training_mode=True,
        render_mode='none'  # Change to 'fast' or 'human' for visual training
    )
    manager = TrainingManager(env)
    return env, manager