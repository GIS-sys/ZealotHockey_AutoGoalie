from tracker_puck import TrackerPuck


def main():
    # Create and start tracker with target FPS
    tracker = TrackerPuck(target_fps=30)  # Adjust FPS as needed
    tracker.start_tracking()

if __name__ == "__main__":
    main()