"""Provide a loading animation while processing a prompt."""
import sys
import time
import threading


def loading_animation(stop_event: threading.Event) -> None:
    """Display a loading animation until the stop event is set."""
    frames = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
    i = 0

    while not stop_event.is_set():
        sys.stdout.write(
            f"\r{frames[i % len(frames)]} Processing prompt ..."
        )
        sys.stdout.flush()
        i += 1
        time.sleep(0.1)
    sys.stdout.write("\r" + " " * 5 + "\r")
    sys.stdout.flush()


def heading() -> None:
    """Display the greeting message."""
    head = """
   \t\t\t\t ██╗    ██╗███████╗██╗      ██████╗   ██████╗  ███╗   ███╗███████╗
   \t\t\t\t ██║    ██║██╔════╝██║     ██╔════╝  ██╔═══██╗ ████╗ ████║██╔════╝
   \t\t\t\t ██║ █╗ ██║█████╗  ██║     ██║       ██║   ██║ ██╔████╔██║█████╗
   \t\t\t\t ██║███╗██║██╔══╝  ██║     ██║       ██║   ██║ ██║╚██╔╝██║██╔══╝
   \t\t\t\t ╚███╔███╔╝███████╗███████╗╚██████╗  ╚██████╔╝ ██║ ╚═╝ ██║███████╗
   \t\t\t\t  ╚══╝╚══╝ ╚══════╝╚══════╝ ╚═════╝   ╚═════╝  ╚═╝     ╚═╝╚══════╝
   """
    print(head)
