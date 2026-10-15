#include "common.h"
#include "sampling.h"
#include "llama.h"

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

static llama_token choose(common_sampler * sampler) {
    llama_token_data data[] = {{10, 2.0f, 0.0f}, {11, 1.0f, 0.0f}};
    llama_token_data_array candidates = {data, 2, -1, false};
    llama_sampler_apply(common_sampler_get(sampler), &candidates);
    if (candidates.selected < 0) {
        throw std::runtime_error("No greedy token selected");
    }
    return candidates.data[candidates.selected].id;
}

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

int main(int argc, char ** argv) {
    if (argc != 2 && argc != 3) {
        return 2;
    }
    auto model_params = llama_model_default_params();
    model_params.vocab_only = true;
    model_params.n_gpu_layers = 0;
    auto * model = llama_model_load_from_file(argv[1], model_params);
    require(model != nullptr, "Could not load vocabulary fixture");
    if (argc == 3) {
        std::ifstream input(argv[2], std::ios::binary);
        require(input.good(), "Could not read tokenizer input");
        std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
        const auto * vocab = llama_model_get_vocab(model);
        const int count = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, false, true);
        std::vector<llama_token> tokens(count);
        const int actual = llama_tokenize(vocab, text.data(), text.size(), tokens.data(), count, false, true);
        require(actual == count, "Tokenizer count mismatch");
        std::printf("[");
        for (int i = 0; i < count; ++i) {
            std::printf("%s%d", i == 0 ? "" : ",", tokens[i]);
        }
        std::puts("]");
        llama_model_free(model);
        return 0;
    }
    common_params_sampling params;
    params.temp = 0.0f;
    params.penalty_repeat = 1.0f;
    params.penalty_freq = 0.0f;
    params.penalty_present = 1.5f;
    params.penalty_last_n = 3584;
    params.samplers = {COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TEMPERATURE};

    auto * original = common_sampler_init(model, params);
    common_sampler_accept(original, 10, false);
    require(choose(original) == 11, "Default sampler behavior changed");
    common_sampler_free(original);

    params.penalties_generated_only = true;
    auto * sampler = common_sampler_init(model, params);
    common_sampler_accept(sampler, 10, false);
    require(choose(sampler) == 10, "Prompt token was incorrectly penalized");
    common_sampler_accept(sampler, 10, true);
    require(choose(sampler) == 11, "Generated token was not penalized");
    for (int i = 0; i < 80; ++i) {
        common_sampler_accept(sampler, 12, true);
    }
    require(choose(sampler) == 11, "Output history was reduced to 64 tokens");
    auto * cloned = common_sampler_clone(sampler);
    require(choose(cloned) == 11, "Cloned sampler lost penalty state");
    common_sampler_reset(cloned);
    common_sampler_accept(cloned, 10, false);
    require(choose(cloned) == 10, "Reset or clone changed generated-only behavior");
    common_sampler_free(cloned);
    common_sampler_free(sampler);
    llama_model_free(model);
    std::puts("PASS: default, prompt exclusion, generated penalty, full history, clone, reset");
    return 0;
}
