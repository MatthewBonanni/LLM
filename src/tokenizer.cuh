#pragma once

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
    public:
        Tokenizer(const std::string& model_path);
        ~Tokenizer();

        int n_vocab() const;

        std::vector<int> tokenize(const std::string& text);
        std::string detokenize(const std::vector<int>& tokens);
        int eos_token_id() const;
        std::string replace_spaces_with_G(const std::string& input);
        std::string replace_G_with_spaces(const std::string& input);

    private:
        void load_pattern_string(const std::string& model_path);
        void load_vocab(const std::string& model_path);
        void load_bpe_merges(const std::string& model_path);
        std::vector<std::string> apply_bpe(const std::vector<std::string>& chars);
        std::vector<std::string> split_utf8_chars(const std::string& input);

        std::string pattern_string;
        std::regex pattern;
        std::unordered_map<std::string, int> token_to_id;
        std::unordered_map<int, std::string> id_to_token;
        std::vector<std::pair<std::string, std::string>> bpe_merges;
};