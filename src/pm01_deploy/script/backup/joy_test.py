import pygame
import time
import sys
import os

def main():
    # Initialize Pygame
    pygame.init()
    pygame.joystick.init()

    # Check for joysticks
    count = pygame.joystick.get_count()
    if count == 0:
        print("Error: No joystick found.")
        return

    # Use the first joystick
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print(f"Initialized Joystick: {joystick.get_name()}")
    print(f"Axes: {joystick.get_numaxes()}")
    print(f"Buttons: {joystick.get_numbuttons()}")
    print(f"Hats: {joystick.get_numhats()}")

    print("\nListening for events... Press Ctrl+C to stop.")

    try:
        while True:
            # Process event queue
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button PRESSED: {event.button}")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Button RELEASED: {event.button}")
                elif event.type == pygame.JOYAXISMOTION:
                    if abs(event.value) > 0.01:
                       print(f"Axis MOTION: {event.axis} Value={event.value:.3f}")
                elif event.type == pygame.JOYHATMOTION:
                     print(f"Hat MOTION: {event.hat} Value={event.value}")

            # Sleep to reduce CPU usage
            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
