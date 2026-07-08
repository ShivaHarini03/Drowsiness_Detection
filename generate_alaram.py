import wave
import struct
import math

sample_rate = 44100
duration = 2
frequency = 900
volume = 16000

with wave.open("alarm.wav", "w") as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(sample_rate)

    for i in range(sample_rate * duration):
        value = int(volume * math.sin(2 * math.pi * frequency * i / sample_rate))
        wav.writeframes(struct.pack("<h", value))

print("alarm.wav created successfully!")