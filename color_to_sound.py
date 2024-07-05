import numpy as np
from scipy.io import wavfile
import cv2

def analyze_image_colors(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    hue_hist = cv2.calcHist([hsv_image], [0], None, [180], [0, 180])
    saturation_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
    value_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
    
    return hue_hist, saturation_hist, value_hist

def color_to_sound(hue_hist, saturation_hist, value_hist, duration=5):
    sample_rate = 44100
    t = np.linspace(0, duration, num=sample_rate * duration, endpoint=False)
    
    frequencies = np.linspace(220, 880, num=180)
    hue_signal = np.zeros_like(t)
    for i, count in enumerate(hue_hist):
        hue_signal += count * np.sin(2 * np.pi * frequencies[i] * t)
    
    saturation_factor = np.sum(saturation_hist * np.arange(256)) / (255 * np.sum(saturation_hist))
    
    value_factor = np.sum(value_hist * np.arange(256)) / (255 * np.sum(value_hist))
    envelope = np.exp(-t / (duration * value_factor))
    
    sound_signal = hue_signal * saturation_factor * envelope
    
    sound_signal = np.int16(sound_signal / np.max(np.abs(sound_signal)) * 32767)
    
    return sample_rate, sound_signal

def generate_sound_from_image(image_path, output_path):
    image = cv2.imread(image_path)
    hue_hist, saturation_hist, value_hist = analyze_image_colors(image)
    sample_rate, sound_signal = color_to_sound(hue_hist, saturation_hist, value_hist)
    wavfile.write(output_path, sample_rate, sound_signal)