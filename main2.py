import phonikud
import time

# calculate how many possible to phonemize per second

count = 0
start_time = time.time()
while True:
    text = 'שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום שלום'
    phonikud.phonemize(text)
    count += 1
    current_time = time.time()
    if current_time - start_time > 5:
        break

print(f"Possible to phonemize per second: {count} texts")