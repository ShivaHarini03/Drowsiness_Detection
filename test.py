import pygame
import time

pygame.mixer.init()
pygame.mixer.music.load("alarm.wav")

print("Playing alarm...")
pygame.mixer.music.play()

time.sleep(3)

pygame.mixer.music.stop()

print("Done!")
