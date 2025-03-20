#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
#include "llm.cuh"

bool ends_with(const std::string& str, const std::string& suffix) {
    return str.size() >= suffix.size() && 
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " <model_path> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --input <file>        Input file with inference texts (plaintext or .h5)" << std::endl;
    std::cout << "  --interactive         Run in interactive mode" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << program_name << " /path/to/model/ --input texts.txt    # Run inference on plaintext file" << std::endl;
    std::cout << "  " << program_name << " /path/to/model/ --input tokens.h5    # Run inference on tokenized .h5 file" << std::endl;
    std::cout << "  " << program_name << " /path/to/model/ --interactive        # Run in interactive mode" << std::endl;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    // Initialize the model with the first argument
    std::string model_path = argv[1];
    LLM model(model_path);
    model.print();
    
    // Parse command line options
    std::string input_file;
    bool interactive_mode = false;
    
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--input" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "--interactive") {
            interactive_mode = true;
        } else {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // If no mode is specified, default to interactive
    if (input_file.empty() && !interactive_mode) {
        interactive_mode = true;
    }
    
    // Process based on the selected mode
    if (!input_file.empty()) {
        bool is_h5_file = ends_with(input_file, ".h5");
        
        if (is_h5_file) {
            // Handle tokenized H5 file
            model.load_tokens_and_run_inference(input_file);
        } else {
            // Handle plaintext file
            std::ifstream file(input_file);
            if (!file.is_open()) {
                throw std::runtime_error("Could not open file: " + input_file);
            }
            
            // Read file line by line and group into texts
            std::vector<std::string> texts;
            std::string line;
            std::string content;
            while (std::getline(file, line)) {
                if (line.empty()) {
                    if (!content.empty()) {
                        texts.push_back(content);
                        content.clear();
                    }
                } else {
                    if (!content.empty()) {
                        content += "\n";
                    }
                    content += line;
                }
            }
            if (!content.empty()) {
                texts.push_back(content);
            }

            model.tokenize_write_and_run_inference(texts);
        }
    } else if (interactive_mode) {
        // Run in interactive mode
        model.run_interactive();
    }
    
    return 0;
}