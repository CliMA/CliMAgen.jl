using GLMakie, HDF5

# run from giorgini2d
# all response data is in the data folder

data_directory = "data/"

response_strings = reshape(["linear_resonse_", "score_response_", "numerical_response_"], 1, :)
label_strings = reshape(["0.3_0.5_0.1_2.0.hdf5", "0.3_0.5_1.0_2.0.hdf5", "0.3_0.5_10.0_2.0.hdf5"], :, 1)

filestrings = data_directory .* response_strings .* label_strings
# filestrings[3,:]