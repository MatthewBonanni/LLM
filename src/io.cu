#include "io.cuh"

#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5_hl.h>

#include "utils.cuh"

void read_dataset(hid_t file_id, const std::string& path, std::vector<fp_t>& data) {
    hsize_t dims[H5S_MAX_RANK];
    int32_t ndims;
    if (H5LTget_dataset_ndims(file_id, path.c_str(), &ndims) < 0) {
        throw std::runtime_error("Error getting dataset dimensions for: " + path);
    }

    if (H5LTget_dataset_info(file_id, path.c_str(), dims, nullptr, nullptr) < 0) {
        throw std::runtime_error("Error getting dataset info for: " + path);
    }

    hsize_t total_size = 1;
    for (int32_t i = 0; i < ndims; ++i) {
        total_size *= dims[i];
    }

    if (data.size() != total_size) {
        throw std::runtime_error("Error: Dataset " + path + " size " + std::to_string(total_size) +
                                 " does not match expected size " + std::to_string(data.size()));
    }

    std::vector<fp_t> temp_data(total_size);
    if (H5LTread_dataset_float(file_id, path.c_str(), temp_data.data()) < 0) {
        throw std::runtime_error("Error reading dataset: " + path);
    }

    // Copy data in row-major order
    std::vector<hsize_t> indices(ndims, 0);
    for (hsize_t i = 0; i < total_size; ++i) {
        hsize_t flat_index = 0;
        hsize_t stride = 1;
        for (int32_t j = ndims - 1; j >= 0; --j) {
            flat_index += indices[j] * stride;
            stride *= dims[j];
        }
        data[i] = temp_data[flat_index];

        // Increment indices
        for (int32_t j = ndims - 1; j >= 0; --j) {
            if (++indices[j] < dims[j]) {
                break;
            }
            indices[j] = 0;
        }
    }
}