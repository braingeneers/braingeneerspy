from example_device import ExampleDevice
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Command line tool for the ExampleDevice utility")
    # Adding arguments with default values and making them optional
    parser.add_argument('--device_name', type=str, required=False, default="spam", help='Name of device (default: spam)')
    parser.add_argument('--eggs', type=int, required=False, default=0, help='Starting quantity of eggs (default: 0)')
    parser.add_argument('--ham', type=int, required=False, default=0, help='Starting quantity of ham (default: 0)')
    parser.add_argument('--spam', type=int, required=False, default=1, help='Starting quantity of spam (default: 1)')

    args = parser.parse_args()

    # Create a device object
    device = ExampleDevice(device_name=args.device_name, eggs=args.eggs, ham=args.ham, spam=args.spam)

    # Start the device activities, running in a loop
    # Control + C should gracefully stop execution
    device.start_mqtt()
