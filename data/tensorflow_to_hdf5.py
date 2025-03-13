import tensorflow as tf
import h5py
import os
import argparse

def convert_checkpoint_to_hdf5(checkpoint_prefix, output_file):
    """
    Load a TensorFlow checkpoint and save all arrays to an HDF5 file

    Args:
        checkpoint_prefix: Path to the checkpoint (without extensions)
        output_file: Path where the HDF5 file should be saved
    """
    print(f"Loading checkpoint from: {checkpoint_prefix}")

    # Use TensorFlow 1.x compatibility mode
    tf1 = tf.compat.v1
    tf1.disable_eager_execution()

    # Create a TensorFlow session
    with tf1.Session() as sess:
        # Load the meta graph and restore the weights
        saver = tf1.train.import_meta_graph(f"{checkpoint_prefix}.meta")
        saver.restore(sess, checkpoint_prefix)

        # Get all variables from the checkpoint
        variables = tf1.global_variables()

        # Create HDF5 file
        with h5py.File(output_file, 'w') as h5f:
            # Add a group for the model weights
            model_group = h5f.create_group("model_weights")

            # Save each variable to the HDF5 file
            for var in variables:
                var_name = var.name.replace(':', '_')  # Replace illegal characters
                var_value = sess.run(var)

                # Create dataset with the variable's value
                model_group.create_dataset(var_name, data=var_value)
                print(f"Saved {var_name} with shape {var_value.shape}")

            # Add metadata about the source checkpoint
            h5f.attrs['source_checkpoint'] = checkpoint_prefix
            h5f.attrs['tensorflow_version'] = tf.__version__

    print(f"Successfully converted checkpoint to HDF5: {output_file}")

def convert_checkpoint_to_hdf5_v2(checkpoint_prefix, output_file):
    """
    Alternative approach for TensorFlow 2.x native checkpoints
    (if the above method doesn't work with your checkpoint format)

    Args:
        checkpoint_prefix: Path to the checkpoint (without extensions)
        output_file: Path where the HDF5 file should be saved
    """
    print(f"Loading checkpoint from: {checkpoint_prefix} using TF2 native method")

    # Load checkpoint using TF2 checkpoint format
    checkpoint = tf.train.Checkpoint()
    status = checkpoint.restore(checkpoint_prefix)

    # Try to ensure the checkpoint is loaded
    status.expect_partial()

    # Get all trainable variables
    # Note: This approach might not get all variables depending on how the model was saved
    checkpoint_vars = {}
    for var in checkpoint.trainable_variables:
        checkpoint_vars[var.name] = var

    # Create HDF5 file
    with h5py.File(output_file, 'w') as h5f:
        # Add a group for the model weights
        model_group = h5f.create_group("model_weights")

        # Save each variable to the HDF5 file
        for var_name, var in checkpoint_vars.items():
            safe_name = var_name.replace(':', '_')  # Replace illegal characters
            var_value = var.numpy()

            # Create dataset with the variable's value
            model_group.create_dataset(safe_name, data=var_value)
            print(f"Saved {safe_name} with shape {var_value.shape}")

        # Add metadata about the source checkpoint
        h5f.attrs['source_checkpoint'] = checkpoint_prefix
        h5f.attrs['tensorflow_version'] = tf.__version__

    print(f"Successfully converted checkpoint to HDF5: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert TensorFlow checkpoint to HDF5')
    parser.add_argument('--checkpoint', type=str, default='model.ckpt',
                        help='Path to the checkpoint file without extensions')
    parser.add_argument('--output', type=str, default='model.h5',
                        help='Path to the output HDF5 file')
    parser.add_argument('--tf2-native', action='store_true',
                        help='Use TF2 native checkpoint loading (for newer formats)')

    args = parser.parse_args()

    if args.tf2_native:
        convert_checkpoint_to_hdf5_v2(args.checkpoint, args.output)
    else:
        convert_checkpoint_to_hdf5(args.checkpoint, args.output)
