import pygame
import tkinter as tk
from tkinter import filedialog

class UserInterface:
    def __init__(self, audio_analyzer, gan, music_generator):
        self.audio_analyzer = audio_analyzer
        self.gan = gan
        self.music_generator = music_generator
        
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        
        self.mode = "visualize"  # "visualize" or "sonify"
    
    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        pos = pygame.mouse.get_pos()
                        self.handle_click(pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_mode()
                    elif event.key == pygame.K_o:
                        self.open_file()
            
            self.update_display()
            pygame.display.flip()
            self.clock.tick(60)
        
        pygame.quit()
    
    def handle_click(self, pos):
        if self.mode == "visualize":
            self.generate_music_from_position(pos)
        else:
            self.generate_sound_from_position(pos)
    
    def toggle_mode(self):
        self.mode = "sonify" if self.mode == "visualize" else "visualize"
    
    def open_file(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        if file_path:
            self.process_file(file_path)
    
    def generate_music_from_position(self, pos):
        # Generate music based on mouse position
        # This is a placeholder implementation
        x, y = pos
        start_string = chr(x % 26 + 65) + chr(y % 26 + 65)
        generated_music = self.music_generator.generate(start_string)
        print(f"Generated music: {generated_music[:50]}...")  # Print first 50 characters
    
    def generate_sound_from_position(self, pos):
        # Generate sound based on mouse position
        # This is a placeholder implementation
        x, y = pos
        hue_hist = np.zeros(180)
        hue_hist[x % 180] = 1
        saturation_hist = np.zeros(256)
        saturation_hist[y % 256] = 1
        value_hist = np.ones(256)
        sample_rate, sound_signal = color_to_sound(hue_hist, saturation_hist, value_hist)
        pygame.sndarray.make_sound(sound_signal).play()
    
    def process_file(self, file_path):
        if file_path.lower().endswith(('.mp3', '.wav')):
            self.process_audio_file(file_path)
        elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            self.process_image_file(file_path)
    
    def process_audio_file(self, file_path):
        features = self.audio_analyzer.extract_audio_features(file_path)
        emotion = self.audio_analyzer.predict(features)
        image = self.gan.generate(features, emotion)
        self.display_image(image)
    
    def process_image_file(self, file_path):
        generate_sound_from_image(file_path, 'output.wav')
        pygame.mixer.music.load('output.wav')
        pygame.mixer.music.play()
    
    def display_image(self, image):
        surface = pygame.surfarray.make_surface(image)
        self.screen.blit(surface, (0, 0))
    
    def update_display(self):
        # Update display based on current mode
        if self.mode == "visualize":
            self.screen.fill((255, 255, 255))  # White background
            font = pygame.font.Font(None, 36)
            text = font.render("Click to generate music", True, (0, 0, 0))
            self.screen.blit(text, (250, 280))
        else:
            self.screen.fill((0, 0, 0))  # Black background
            font = pygame.font.Font(None, 36)
            text = font.render("Click to generate sound", True, (255, 255, 255))
            self.screen.blit(text, (250, 280))