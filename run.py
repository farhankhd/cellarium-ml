import subprocess

# Define the base command and arguments
command = [
    "cellarium-ml", "llomics", "fit"
]

# Define the data filenames URL
url = "https://storage.googleapis.com/dsp-cellarium-cas-public/cas_50m_homo_sapiens_no_cancer_extract/extract_0.h5ad"

# Add the rest of the arguments
args = [
    "--data.filenames", url,
    "--data.shard_size", "10000",
    "--data.max_cache_size", "2",
    "--data.batch_size", "5",
    "--data.num_workers", "1",
    "--trainer.accelerator", "cpu",
    "--trainer.devices", "1",
    "--trainer.default_root_dir", "runs/llomics",
    "--trainer.max_steps", "10"
]

command.extend(args)

# Execute the command
subprocess.run(command)
