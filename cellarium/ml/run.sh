cellarium-ml llomics fit \
    --data.filenames "https://storage.googleapis.com/dsp-cellarium-cas-public/cas_50m_homo_sapiens_no_cancer_extract/extract_{1..2}.h5ad" \
    --data.shard_size 10000 \
    --data.max_cache_size 2 \
    --data.batch_size 5 \
    --data.num_workers 1 \
    --trainer.accelerator cpu \
    --trainer.devices 1 \
    --trainer.default_root_dir runs/llomics \
    --trainer.max_steps 10

#gs://dsp-cellarium-cas-public/test-data/test_{0..3}.h5ad