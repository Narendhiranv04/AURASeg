import time
import torch
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser

# Function to compute speed of the model on CPU
def compute_speed(model, input_size, iteration=100):
    model.eval()
    model = model.cpu()  # Ensure the model is on the CPU

    # Create a dummy input tensor on CPU
    input_tensor = torch.randn(*input_size)

    # Warm-up runs
    for _ in range(50):
        model(input_tensor)

    print('=========Speed Testing=========')
    t_start = time.time()

    for _ in range(iteration):
        model(input_tensor)

    elapsed_time = time.time() - t_start

    speed_time = (elapsed_time / iteration) * 1000  # Speed per iteration in ms
    fps = iteration / elapsed_time  # Frames per second

    print(f'Elapsed Time: {elapsed_time:.2f} s for {iteration} iterations')
    print(f'Speed Time: {speed_time:.2f} ms / iter   FPS: {fps:.2f}')
    
    return speed_time, fps

# Main function to run the evaluation
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--size", type=str, default="512,1024", help="Input size of the model (height, width)")
    parser.add_argument('--num-channels', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--iter', type=int, default=100)
    
    args = parser.parse_args()

    # Parse the input size
    h, w = map(int, args.size.split(','))

    # Assume you have an FPN model already built
    from model import FullModelWithAPF  # Import your model definition

    # Initialize your model
    model = FullModelWithAPF()  # Replace with your actual FPN model if needed

    # Evaluate the model speed on CPU
    compute_speed(model, (args.batch_size, args.num_channels, h, w), iteration=args.iter)

