import numpy as np

class FlappyBirdEnv:
    """
    Simplified Flappy Bird environment
    State: [bird_y, bird_velocity, next_pipe_x, next_pipe_y]
    Action: 0 (do nothing) or 1 (flap)
    """
    def __init__(self):
        # Physics parameters
        self.gravity = 0.5
        self.flap_strength = -8
        self.velocity_cap = 10
        self.pipe_velocity = -2
        
        # Game parameters
        self.width = 400
        self.height = 400
        self.bird_x = 100  # fixed x position
        self.pipe_width = 50
        self.gap_height = 100
        
        # Initialize
        self.reset()

    def reset(self):
        self.bird_y = self.height / 2
        self.bird_velocity = 0
        self.score = 0
        self.steps = 0
        self.done = False
        
        # Initialize first pipe
        self.pipe_x = self.width
        self.pipe_gap_y = np.random.randint(100, self.height - 100)
        
        return self._get_state()

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True, {}
        
        reward = 0.1  # small reward for surviving
        self.steps += 1
        
        # Apply action and gravity
        if action == 1:  # flap
            self.bird_velocity = self.flap_strength
        self.bird_velocity = min(self.bird_velocity + self.gravity, self.velocity_cap)
        self.bird_y += self.bird_velocity
        
        # Move pipe
        self.pipe_x += self.pipe_velocity
        
        # Generate new pipe if current one is off screen
        if self.pipe_x < -self.pipe_width:
            self.pipe_x = self.width
            self.pipe_gap_y = np.random.randint(100, self.height - 100)
            self.score += 1
            reward = 1.0  # reward for passing pipe
        
        # Check for collisions
        if self._check_collision():
            reward = -1
            self.done = True
        
        # Check if bird is out of bounds
        if self.bird_y < 0 or self.bird_y > self.height:
            reward = -1
            self.done = True
        
        return self._get_state(), reward, self.done, {"score": self.score}

    def _check_collision(self):
        # Check if bird collides with pipes
        if (self.bird_x + 20 > self.pipe_x and 
            self.bird_x - 20 < self.pipe_x + self.pipe_width):
            if (self.bird_y < self.pipe_gap_y - self.gap_height/2 or 
                self.bird_y > self.pipe_gap_y + self.gap_height/2):
                return True
        return False

    def _get_state(self):
        return np.array([
            self.bird_y / self.height,  # normalize to [0,1]
            self.bird_velocity / self.velocity_cap,
            (self.pipe_x - self.bird_x) / self.width,
            self.pipe_gap_y / self.height
        ], dtype=np.float32)

    def render(self):
        # Simple ASCII rendering
        display = [[' ' for _ in range(80)] for _ in range(20)]
        
        # Scale positions to display size
        bird_display_y = int(self.bird_y / self.height * 19)
        bird_display_x = int(self.bird_x / self.width * 79)
        pipe_display_x = int(self.pipe_x / self.width * 79)
        gap_display_y = int(self.pipe_gap_y / self.height * 19)
        
        # Draw pipes
        if 0 <= pipe_display_x < 80:
            for y in range(20):
                if abs(y - gap_display_y) > self.gap_height/self.height * 10:
                    display[y][pipe_display_x] = '|'
        
        # Draw bird
        if 0 <= bird_display_y < 20 and 0 <= bird_display_x < 80:
            display[bird_display_y][bird_display_x] = '>'
        
        # Print display
        print('\n'.join([''.join(row) for row in display]))
        print(f"Score: {self.score}")
        print()
