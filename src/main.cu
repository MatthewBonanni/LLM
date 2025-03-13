#include <iostream>
#include <stdexcept>

#include "llm.cuh"

int main(int argc, char **argv) {
    // Get the path to the model from the command line
    if (argc != 2) {
        throw std::runtime_error("Usage: " + std::string(argv[0]) + " <path-to-model>");
    }

    // Load the model
    LLM model(argv[1]);
    model.print();
    model.run_interactive();
}