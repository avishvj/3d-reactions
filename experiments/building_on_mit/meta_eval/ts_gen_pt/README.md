## PyTorch port of MIT's ts_gen model https://github.com/PattanaikL/ts_gen

- Data processing handled in 3d-reactions/ts_vae/data_processors/ts_gen_processor.py.
- Edge features in MIT model are represented with dim of batch_size x N x N x 3. For easier batching, we represent it as dimensions of batch_size * N * N x 3 and reshape when necessary.
- Running the model and plotting results is possible via 3d-reactions/ts_gen_port.ipynb. Currently, the plots are not identical to the ts_gen model so something is wrong.
