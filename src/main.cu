#include <fstream>
#include <iostream>
#include <stdexcept>

#include "llm.cuh"

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [input_file]" << std::endl;
        return 1;
    }

    // Initialize the model
    LLM model(argv[1]);
    model.print();

    // Check if a file is provided as an argument
    if (argc == 3) {
        // Run inference on the provided file
        std::ifstream input_file(argv[2]);
        if (!input_file.is_open()) {
            throw std::runtime_error("Could not open file: " + std::string(argv[1]));
        }

        std::string content((std::istreambuf_iterator<char>(input_file)),
                            (std::istreambuf_iterator<char>()          ));
        std::cout << "Input text: " << content << std::endl;
        model.run_inference(content);
    } else {
        model.run_interactive();
    }
}