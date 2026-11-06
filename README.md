# MDSS GPU Decoder

This repository contains an impelmentation of the MDSS decoding algorithm from [here](https://eprint.iacr.org/2026/747) where the weak Popov form calculations are parallelized on a GPU.

It contains a Dockerfile that builds a container with the compiled decoder. The executable can be found at `/workspace/gpu_decoder/build`.

## Building

1. Install the [NTL](https://libntl.org/) library, version 11.5.1.
1. Install the [GMP](https://gmplib.org/) library, version 6.3.0.
1. `cd gpu_decoder`
1. `mkdir build`
1. `cd build`
1. `cmake ..`
1. `make`

## Running the Decoder

To run the decoder you must have an instance and a configuration file to point it at.

### Configuration Files

An example configuration file is below.

```json
{
  "threshold": 100,
  "chunk_size": 256,
  "termination_condition": "FIRST_FAILURE",
  "step_parameters": [
    { "batch_size": 1000, "c": 25 },
    { "batch_size": 2000, "c": 50 }
  ]
}
```

The config has the following fields:

- `threshold`: The threshold at which a value should be recovered. If there are at least this many points in the instance that agree with each other, their value will be returned.
- `chunk_size`: Determines the amount of GPU memory used concurrently when computing the lagrange interpolations. The larger the value the faster the computation while the more memory it requires. When `n` is the total number of points in the instance you should set it so that `4 * n * chunk_size` is decently less than your total available VRAM.
- `termination_condition`: When the decoder should stop running. Available options are:
  - `FIRST_FAILURE`: Terminate after any step fails to recover a value. Note that this setting loops on the final step parameters until failure.
  - `ALL_STEPS_DONE`: Terminate after all steps have completed.
  - `ALL_STEPS_DONE_OR_FIRST_FAILURE`: Terminate when either of the two above conditions is satisfied.
- `step_parameters`: Configurations for each decode attempt. Each contains a `batch_size` which is the number of points it will sample from the full instance as well as a `c` value, the number of polynomial evaluations it will use to decode. If you would like the `batch_size` to always be equal to the total number of remaining points, set it higher than the number of points in your instance. When `batch_size` is larger than the number of points remaining decode will be run over all remaining points.

### Instances
