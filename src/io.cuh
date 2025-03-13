#pragma once

#include <string>
#include <vector>

#include <hdf5_hl.h>

void read_dataset(hid_t file_id, const std::string& path, std::vector<float>& data);