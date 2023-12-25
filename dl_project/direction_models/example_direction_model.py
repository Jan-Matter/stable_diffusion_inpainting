import torch
from direction_model import DirectionModel  # Assuming DirectionModel is saved in direction_model.py

def main():
    # Parameters
    batch_size = 5  # Number of vectors in the batch
    c_length = 10  # Length of each vector
    direction_count = 7  # Number of directions
    depth = 2  # Depth of MLPs in DirectionModel

    # Generate a batch of arbitrary vectors
    x = torch.randn(batch_size, c_length)
    print("Original Input Tensor:\n", x)

    # Instantiate the DirectionModel
    model = DirectionModel(
        direction_count=direction_count,
        c_length=c_length,
        depth=depth,
        alpha=0.1,
        normalize=True,
        bias=True,
        batchnorm=True,
        final_norm=False
    )

    # Run the batch through the model using the regular forward method
    output = model(x)

    # Print the output from the regular forward method
    print("Output Shape:", output.shape)
    print("Output Tensor:\n", output)

    # Calculate the approximate base vector from the regular forward method
    approximate_base = torch.mean(output, dim=1)
    print("Approximate Base Vectors from forward method:\n", approximate_base)

    # Test the forward_single method for each direction
    for direction_index in range(direction_count):
        output_single = model.forward_single(x, direction_index)

        # Print the output shape and tensor for each direction
        print(f"Output Shape for direction {direction_index}:", output_single.shape)
        print(f"Output Tensor for direction {direction_index}:\n", output_single)

if __name__ == "__main__":
    main()
