import sys
import time
import threading


def loading_animation(stop_event):
    #frames = ["⏳", "🔄", "⌛", "🔄"]
    frames = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
    i = 0

    while not stop_event.is_set():
        sys.stdout.write(
            f"\r{frames[i % len(frames)]} Processing prompt ..."
        )
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 40 + "\r")
    sys.stdout.flush()