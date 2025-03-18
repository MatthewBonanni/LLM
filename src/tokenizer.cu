#include "tokenizer.cuh"

#include <iostream>
#include <fstream>
#include <sstream>

#include <nlohmann/json.hpp>

Tokenizer::Tokenizer(const std::string& model_path) {
    load_pattern_string(model_path);
    load_vocab(model_path);
    load_bpe_merges(model_path);
}

Tokenizer::~Tokenizer() {
}

void Tokenizer::load_pattern_string(const std::string& model_path) {
    std::ifstream file(model_path + "/pattern_string");
    if (!file) {
        throw std::runtime_error("Error: Cannot open pattern_string");
    }

    std::getline(file, pattern_string);
    file.close();

    if (pattern_string.empty()) {
        throw std::runtime_error("Error: pattern_string is empty");
    }

    pattern = std::regex(pattern_string);
}

void Tokenizer::load_vocab(const std::string& model_path) {
    std::ifstream file(model_path + "/encoder.json");
    if (!file) {
        throw std::runtime_error("Error: Cannot open encoder.json");
    }

    nlohmann::json j;
    file >> j;
    for (auto& [token, id] : j.items()) {
        token_to_id[token] = id;
        id_to_token[id] = token;
    }

    std::cout << "Loaded " << token_to_id.size() << " tokens from encoder.json\n";

    if (token_to_id.size() == 0) {
        throw std::runtime_error("No tokens loaded from encoder.json");
    }
}

void Tokenizer::load_bpe_merges(const std::string& model_path) {
    std::ifstream file(model_path + "/vocab.bpe");
    if (!file.is_open()) {
        throw std::runtime_error("Error: Cannot open vocab.bpe");
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        std::string first, second;
        if (!(iss >> first >> second)) {
            continue;
        }

        bpe_merges.push_back({first, second});
    }
    file.close();
}

int Tokenizer::n_vocab() const {
    return token_to_id.size();
}

std::vector<int> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::vector<int> token_ids;

    auto words_begin = std::sregex_iterator(text.begin(),
                                            text.end(),
                                            pattern);
    auto words_end = std::sregex_iterator();

    std::vector<std::string> words;
    std::string word;
    for (auto it = words_begin; it != words_end; ++it) {
        word = replace_spaces_with_G(it->str());
        words.push_back(word);
    }

    for (const std::string& word : words) {
        // Split word into characters
        std::vector<std::string> chars = split_utf8_chars(word);

        // Apply BPE merges to characters
        std::vector<std::string> word_tokens = apply_bpe(chars);
        tokens.insert(tokens.end(), word_tokens.begin(), word_tokens.end());
    }

    // Convert merged tokens to IDs
    for (const auto& token : tokens) {
        if (token_to_id.find(token) != token_to_id.end()) {
            token_ids.push_back(token_to_id[token]);
        } else {
            std::cerr << "Warning: Unknown token '" << token << "'\n";
        }
    }

    return token_ids;
}

std::string Tokenizer::detokenize(const std::vector<int>& tokens) {
    std::string text;
    for (int token_id : tokens) {
        if (id_to_token.find(token_id) != id_to_token.end()) {
            text += id_to_token[token_id];
        } else {
            std::cerr << "Warning: Unknown token ID '" << token_id << "'\n";
        }
    }
    return text;
}

std::vector<std::string> Tokenizer::apply_bpe(const std::vector<std::string>& chars) {
    std::vector<std::string> tokens = chars;
    std::string token;

    while (tokens.size() > 1) {
        std::pair<std::string, std::string> best_pair;
        int best_pair_rank = -1;
        int best_pair_index = -1;

        // Iterate through all pairs of tokens
        for (size_t i = 0; i < tokens.size() - 1; i++) {
            std::pair<std::string, std::string> pair = {tokens[i], tokens[i + 1]};
            auto it = std::find(bpe_merges.begin(), bpe_merges.end(), pair);
            if (it != bpe_merges.end()) {
                int rank = std::distance(bpe_merges.begin(), it);
                if (rank < best_pair_rank || best_pair_rank == -1) {
                    best_pair = pair;
                    best_pair_rank = rank;
                    best_pair_index = i;
                }
            }
        }

        if (best_pair_index == -1) {
            break;
        }

        // Merge the best pair
        std::string merged = best_pair.first + best_pair.second;
        tokens[best_pair_index] = merged;
        tokens.erase(tokens.begin() + best_pair_index + 1);
    }

    return tokens;
}

std::string Tokenizer::replace_spaces_with_G(const std::string& input) {
    std::string result;
    for (char c : input) {
        if (c == ' ') {
            result += "Ġ";
        } else {
            result += c;
        }
    }
    return result;
}

std::vector<std::string> Tokenizer::split_utf8_chars(const std::string& input) {
    std::vector<std::string> chars;
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
    std::u32string utf32 = converter.from_bytes(input);

    for (char32_t c : utf32) {
        chars.push_back(converter.to_bytes(c));
    }
    return chars;
}

int Tokenizer::eos_token_id() const {
    return token_to_id.at("<|endoftext|>");
}