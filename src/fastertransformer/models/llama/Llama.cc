/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/models/llama/Llama.h"
#include "src/fastertransformer/kernels/bert_preprocess_kernels.h"
#include "src/fastertransformer/kernels/decoding_kernels.h"
#include "src/fastertransformer/kernels/gpt_kernels.h"
#include "src/fastertransformer/layers/beam_search_layers/BaseBeamSearchLayer.h"
#include <algorithm>

namespace fastertransformer {

template<typename T>
void Llama<T>::initialize()
{
    gpt_context_decoder_ = new LlamaContextDecoder<T>(head_num_,
                                                      size_per_head_,
                                                      inter_size_,
                                                      num_layer_,
                                                      rotary_embedding_dim_,
                                                      neox_rotary_style_,
                                                      use_gptj_residual_,
                                                      layernorm_eps_,
                                                      tensor_para_,
                                                      pipeline_para_,
                                                      stream_,
                                                      cublas_wrapper_,
                                                      allocator_,
                                                      is_free_buffer_after_forward_,
                                                      is_context_qk_buf_float_,
                                                      attention_type_,
                                                      int8_mode_,
                                                      custom_all_reduce_comm_,
                                                      enable_custom_all_reduce_);

    gpt_decoder_ = new LlamaDecoder<T>(head_num_,
                                       size_per_head_,
                                       inter_size_,
                                       num_layer_,
                                       rotary_embedding_dim_,
                                       neox_rotary_style_,
                                       use_gptj_residual_,
                                       layernorm_eps_,
                                       tensor_para_,
                                       pipeline_para_,
                                       stream_,
                                       cublas_wrapper_,
                                       allocator_,
                                       is_free_buffer_after_forward_,
                                       int8_mode_,
                                       custom_all_reduce_comm_,
                                       enable_custom_all_reduce_);

    dynamic_decode_layer_ = new DynamicDecodeLayer<float>(vocab_size_,
                                                          vocab_size_padded_,
                                                          0,  // end_id, deprecated
                                                          stream_,
                                                          cublas_wrapper_,
                                                          allocator_,
                                                          is_free_buffer_after_forward_,
                                                          cuda_device_prop_);

    // parse env overrides
    if (std::getenv("LLAMA_STREAM_CB_STEP") != nullptr) {
        try {
            int callback_step_from_env = stoi(std::string(std::getenv("LLAMA_STREAM_CB_STEP")));
            token_generated_cb_step_   = callback_step_from_env;
            FT_LOG_INFO("Override stream callback step to %d from LLAMA_STREAM_CB_STEP", token_generated_cb_step_);
        }
        catch (...) {
            FT_LOG_WARNING("convert LLAMA_STREAM_CB_STEP err, use default value %d", token_generated_cb_step_);
        }
    }
}

template<typename T>
void Llama<T>::allocateBuffer()
{
    FT_CHECK(false);
}

template<typename T>
void Llama<T>::allocateBuffer(
    size_t batch_size, size_t beam_width, size_t max_seq_len, size_t max_cache_seq_len, size_t max_input_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    const size_t batchxbeam      = batch_size * beam_width;
    const size_t self_cache_size = (num_layer_ / pipeline_para_.world_size_) * batchxbeam * max_cache_seq_len
                                   * hidden_units_ / tensor_para_.world_size_;

    if (vocab_size_ != vocab_size_padded_) {
        padded_embedding_kernel_ =
            (T*)(allocator_->reMalloc(padded_embedding_kernel_, sizeof(T) * hidden_units_ * vocab_size_padded_, true));
        padded_embedding_kernel_ptr_ = padded_embedding_kernel_;
    }

    input_attention_mask_ = (T*)(allocator_->reMalloc(
        input_attention_mask_, sizeof(T) * batchxbeam * max_seq_len * max_cache_seq_len, false));
    decoder_input_buf_ = (T*)(allocator_->reMalloc(decoder_input_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    decoder_output_buf_ =
        (T*)(allocator_->reMalloc(decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    normed_decoder_output_buf_ =
        (T*)(allocator_->reMalloc(normed_decoder_output_buf_, sizeof(T) * batchxbeam * hidden_units_, false));
    logits_buf_ = (float*)(allocator_->reMalloc(logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    nccl_logits_buf_ =
        (float*)(allocator_->reMalloc(nccl_logits_buf_, sizeof(float) * batchxbeam * vocab_size_padded_, false));
    cum_log_probs_    = (float*)(allocator_->reMalloc(cum_log_probs_, sizeof(float) * batchxbeam, false));
    finished_buf_     = (bool*)(allocator_->reMalloc(finished_buf_, sizeof(bool) * batchxbeam, false));
    h_finished_buf_   = new bool[batchxbeam];
    sequence_lengths_ = (int*)(allocator_->reMalloc(sequence_lengths_, sizeof(int) * batchxbeam, false));

    key_cache_   = (T*)(allocator_->reMalloc(key_cache_, sizeof(T) * self_cache_size * 2, true));
    value_cache_ = key_cache_ + self_cache_size;
    if (beam_width > 1) {
        cache_indirections_[0] =
            (int*)(allocator_->reMalloc(cache_indirections_[0], sizeof(int) * batchxbeam * max_seq_len * 2, true));
        cache_indirections_[1] = cache_indirections_[0] + batchxbeam * max_seq_len;
    }

    // prompt_learning weight batch ptrs
    prompt_learning_weight_batch_ =
        (const T**)(allocator_->reMalloc(prompt_learning_weight_batch_, sizeof(T*) * batchxbeam, false));
    tiled_prompt_lengths_buf_ =
        (int*)(allocator_->reMalloc(tiled_prompt_lengths_buf_, sizeof(int) * batchxbeam, false));

    tiled_input_ids_buf_ =
        (int*)(allocator_->reMalloc(tiled_input_ids_buf_, sizeof(int) * batchxbeam * max_input_len, true));
    tiled_input_lengths_buf_ = (int*)(allocator_->reMalloc(tiled_input_lengths_buf_, sizeof(int) * batchxbeam, true));
    tiled_total_padding_count_ =
        (int*)allocator_->reMalloc(tiled_total_padding_count_, batchxbeam * sizeof(int), false);

    transposed_output_ids_buf_ =
        (int*)(allocator_->reMalloc(transposed_output_ids_buf_, sizeof(int) * batchxbeam * max_seq_len, true));
    output_ids_buf_ = (int*)(allocator_->reMalloc(output_ids_buf_, sizeof(int) * batchxbeam * max_seq_len, true));
    parent_ids_buf_ = (int*)(allocator_->reMalloc(parent_ids_buf_, sizeof(int) * batchxbeam * max_seq_len, true));
    seq_limit_len_  = (uint32_t*)(allocator_->reMalloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false));
    masked_tokens_ = (bool*)(allocator_->reMalloc(masked_tokens_, sizeof(bool) * batchxbeam * max_cache_seq_len, true));

    start_ids_buf_ = (int*)(allocator_->reMalloc(start_ids_buf_, sizeof(int) * batch_size, false));
    end_ids_buf_   = (int*)(allocator_->reMalloc(end_ids_buf_, sizeof(int) * batch_size, false));

    context_decoder_input_buf_  = (T*)(allocator_->reMalloc(
        context_decoder_input_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    context_decoder_output_buf_ = (T*)(allocator_->reMalloc(
        context_decoder_output_buf_, sizeof(T) * batchxbeam * max_input_len * hidden_units_, false));
    output_log_probs_buf_ =
        (float*)(allocator_->reMalloc(output_log_probs_buf_, sizeof(float) * batchxbeam * max_seq_len, false));

    generation_should_stop_ = (bool*)allocator_->reMalloc(generation_should_stop_, sizeof(bool), true, true);

    if (shared_contexts_ratio_ > 0.0f) {
        shared_contexts_idx_  = (int*)allocator_->reMalloc(shared_contexts_idx_, batch_size * sizeof(int), false);
        batch_to_compact_idx_ = (int*)allocator_->reMalloc(batch_to_compact_idx_, batchxbeam * sizeof(int), false);
        compact_idx_          = (int*)allocator_->reMalloc(compact_idx_, batch_size * sizeof(int), false);
        compact_size_         = (int*)allocator_->reMalloc(compact_size_, sizeof(int), false);
    }

    is_allocate_buffer_ = true;
}

template<typename T>
void Llama<T>::freeBuffer()
{
    if (is_allocate_buffer_) {
        if (vocab_size_ != vocab_size_padded_) {
            padded_embedding_kernel_ptr_ = nullptr;
            allocator_->free((void**)(&padded_embedding_kernel_));
        }

        allocator_->free((void**)(&input_attention_mask_));
        allocator_->free((void**)(&decoder_input_buf_));
        allocator_->free((void**)(&decoder_output_buf_));
        allocator_->free((void**)(&normed_decoder_output_buf_));
        allocator_->free((void**)(&logits_buf_));
        allocator_->free((void**)(&nccl_logits_buf_));
        allocator_->free((void**)(&cum_log_probs_));
        allocator_->free((void**)(&finished_buf_));
        delete[] h_finished_buf_;
        allocator_->free((void**)(&sequence_lengths_));

        allocator_->free((void**)(&key_cache_));
        if (cache_indirections_[0] != nullptr) {
            allocator_->free((void**)(&cache_indirections_)[0]);
        }

        allocator_->free((void**)(&prompt_learning_weight_batch_));
        allocator_->free((void**)(&tiled_prompt_lengths_buf_));

        allocator_->free((void**)(&tiled_input_ids_buf_));
        allocator_->free((void**)(&tiled_input_lengths_buf_));
        allocator_->free((void**)(&tiled_total_padding_count_));

        allocator_->free((void**)(&transposed_output_ids_buf_));
        allocator_->free((void**)(&output_ids_buf_));
        allocator_->free((void**)(&parent_ids_buf_));
        allocator_->free((void**)(&seq_limit_len_));
        allocator_->free((void**)(&masked_tokens_));

        allocator_->free((void**)(&start_ids_buf_));
        allocator_->free((void**)(&end_ids_buf_));

        allocator_->free((void**)(&context_decoder_input_buf_));
        allocator_->free((void**)(&context_decoder_output_buf_));
        allocator_->free((void**)(&output_log_probs_buf_));

        allocator_->free((void**)(&generation_should_stop_), true);

        if (shared_contexts_ratio_ > 0.0f) {
            allocator_->free((void**)(&shared_contexts_idx_));
            allocator_->free((void**)(&compact_size_));
        }

        is_allocate_buffer_ = false;
    }
}

template<typename T>
Llama<T>::Llama(size_t                              head_num,
                size_t                              size_per_head,
                size_t                              inter_size,
                size_t                              num_layer,
                size_t                              vocab_size,
                size_t                              rotary_embedding_dim,
                float                               layernorm_eps,
                int                                 start_id,
                int                                 end_id,
                int                                 prompt_learning_start_id,  // only needed by p/prompt-tuning
                PromptLearningType                  prompt_learning_type,
                bool                                use_gptj_residual,
                float                               beam_search_diversity_rate,
                size_t                              top_k,
                float                               top_p,
                unsigned long long                  random_seed,
                float                               temperature,
                float                               len_penalty,
                float                               repetition_penalty,
                cudaStream_t                        stream,
                cublasMMWrapper*                    cublas_wrapper,
                IAllocator*                         allocator,
                bool                                is_free_buffer_after_forward,
                cudaDeviceProp*                     cuda_device_prop,
                AttentionType                       attention_type,
                int                                 int8_mode,
                std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                int                                 enable_custom_all_reduce,
                float                               shared_contexts_ratio):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    layernorm_eps_(layernorm_eps),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    use_gptj_residual_(use_gptj_residual),
    hidden_units_(head_num * size_per_head),
    local_head_num_(head_num / 1),
    attention_type_(attention_type),
    int8_mode_(int8_mode),
    shared_contexts_ratio_(shared_contexts_ratio)
{
    tensor_para_.world_size_   = 1;
    tensor_para_.rank_         = 0;
    pipeline_para_.world_size_ = 1;
    pipeline_para_.rank_       = 0;

    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value
#ifdef ENABLE_BF16
        || std::is_same<__nv_bfloat16, T>::value
#endif
    ) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    initialize();
}

template<typename T>
Llama<T>::Llama(size_t                              head_num,
                size_t                              size_per_head,
                size_t                              inter_size,
                size_t                              num_layer,
                size_t                              vocab_size,
                size_t                              rotary_embedding_dim,
                float                               layernorm_eps,
                int                                 start_id,
                int                                 end_id,
                int                                 prompt_learning_start_id,  // only needed by p/prompt-tuning
                PromptLearningType                  prompt_learning_type,
                bool                                use_gptj_residual,
                float                               beam_search_diversity_rate,
                size_t                              top_k,
                float                               top_p,
                unsigned long long                  random_seed,
                float                               temperature,
                float                               len_penalty,
                float                               repetition_penalty,
                NcclParam                           tensor_para,
                NcclParam                           pipeline_para,
                cudaStream_t                        stream,
                cublasMMWrapper*                    cublas_wrapper,
                IAllocator*                         allocator,
                bool                                is_free_buffer_after_forward,
                cudaDeviceProp*                     cuda_device_prop,
                AttentionType                       attention_type,
                int                                 int8_mode,
                std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm,
                int                                 enable_custom_all_reduce,
                float                               shared_contexts_ratio):
    BaseLayer(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num_(head_num),
    size_per_head_(size_per_head),
    inter_size_(inter_size),
    num_layer_(num_layer),
    vocab_size_(vocab_size),
    rotary_embedding_dim_(rotary_embedding_dim),
    layernorm_eps_(layernorm_eps),
    start_id_(start_id),
    end_id_(end_id),
    prompt_learning_start_id_(prompt_learning_start_id),
    prompt_learning_type_(prompt_learning_type),
    use_gptj_residual_(use_gptj_residual),
    hidden_units_(head_num * size_per_head),
    tensor_para_(tensor_para),
    pipeline_para_(pipeline_para),
    local_head_num_(head_num / tensor_para.world_size_),
    custom_all_reduce_comm_(custom_all_reduce_comm),
    enable_custom_all_reduce_(enable_custom_all_reduce),
    attention_type_(attention_type),
    int8_mode_(int8_mode),
    shared_contexts_ratio_(shared_contexts_ratio)
{
    int local_vacab_size = ceil(vocab_size_ / 1.f / tensor_para_.world_size_);
    if (std::is_same<half, T>::value) {
        local_vacab_size = ceil(local_vacab_size / 8.f) * 8;
    }
    vocab_size_padded_ = (size_t)local_vacab_size * tensor_para_.world_size_;
    initialize();
}

template<typename T>
Llama<T>::Llama(Llama<T> const& llama):
    BaseLayer(llama),
    head_num_(llama.head_num_),
    size_per_head_(llama.size_per_head_),
    inter_size_(llama.inter_size_),
    num_layer_(llama.num_layer_),
    vocab_size_(llama.vocab_size_),
    rotary_embedding_dim_(llama.rotary_embedding_dim_),
    layernorm_eps_(llama.layernorm_eps_),
    start_id_(llama.start_id_),
    end_id_(llama.end_id_),
    prompt_learning_start_id_(llama.prompt_learning_start_id_),
    prompt_learning_type_(llama.prompt_learning_type_),
    use_gptj_residual_(llama.use_gptj_residual_),
    hidden_units_(llama.hidden_units_),
    tensor_para_(llama.tensor_para_),
    pipeline_para_(llama.pipeline_para_),
    local_head_num_(llama.local_head_num_),
    vocab_size_padded_(llama.vocab_size_padded_),
    custom_all_reduce_comm_(llama.custom_all_reduce_comm_),
    enable_custom_all_reduce_(llama.enable_custom_all_reduce_),
    attention_type_(llama.attention_type_),
    int8_mode_(llama.int8_mode_),
    shared_contexts_ratio_(llama.shared_contexts_ratio_)
{
    initialize();
}

template<typename T>
Llama<T>::~Llama()
{
    delete gpt_decoder_;
    delete dynamic_decode_layer_;
    delete gpt_context_decoder_;
    freeBuffer();
}

template<typename T>
void Llama<T>::registerCallback(callback_sig* fn, void* ctx)
{
    token_generated_cb_  = fn;
    token_generated_ctx_ = ctx;
}

template<typename T>
void Llama<T>::unRegisterCallback()
{
    token_generated_cb_  = nullptr;
    token_generated_ctx_ = nullptr;
}

template<typename T>
void Llama<T>::forward(std::vector<Tensor>*       output_tensors,
                       const std::vector<Tensor>* input_tensors,
                       const LlamaWeight<T>*      gpt_weights)
{
    FT_CHECK(false);
}

template<typename T>
void Llama<T>::forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                       const std::unordered_map<std::string, Tensor>* input_tensors,
                       const LlamaWeight<T>*                          gpt_weights)
{
    // input_tensors:
    //      input_ids [batch_size, max_input_length]
    //      input_lengths [batch_size]
    //      prompt_learning_task_name_ids [batch_size] on cpu, optional
    //      output_seq_len [batch_size] on cpu
    //      start_id [batch_size] on cpu, optional
    //      end_id [batch_size] on cpu, optional
    //      stop_words_list [batch_size, 2, stop_words_length], optional
    //      bad_words_list [2, bad_words_length] or [batch_size, 2, bad_words_length], optional
    //      runtime_top_k [1] or [batch_size] on cpu, optional, uint.
    //      runtime_top_p [1] or [batch_size] on cpu, optional, float.
    //      beam_search_diversity_rate [1] or [batch_size] on cpu, optional, float.
    //      temperature [1] or [batch_size] on cpu, optional, float.
    //      len_penalty [1] or [batch_size] on cpu, optional, float.
    //      repetition_penalty [1] or [batch_size] on cpu, optional, float.
    //      min_length [1] or [batch_size] on cpu, optional, int
    //      random_seed [1] or [batch_size] on cpu, optional, unsigned long long int.
    //      request_prompt_lengths [batch_size], optional
    //      request_prompt_embedding [batch_size, max_prompt_length, hidden_units], float, optional
    //      requst_prompt_type [batch_size], int, optional
    //      top_p_decay [batch_size] on gpu, float, optional
    //      top_p_min [batch_size] on gpu, float, optional
    //      top_p_reset_ids [batch_size] on gpu, uint32, optional

    // output_tensors:
    //      output_ids [batch_size, beam_width, max_output_seq_len]
    //      sequence_length [batch_size, beam_width]
    //      output_log_probs [batch_size, beam_width, request_output_seq_len], must be float*.
    //          optional. It leads to additional computing cost. If we don't need this result, don't put it.
    //      cum_log_probs [batch_size, beam], optional, must be float*.
    //          optional. It leads to additional computing cost. If we don't need this result, don't put it.

    // Step is from max_input_length ~ max_output_seq_len,
    // When step = k,  we put output ids and caches at step k, and the sequence_length would be k - 1 before
    // complete this step.
    // When there is no input_ids, put the start token at step 0 of output_ids_buf_. After forward, only copy
    // the step 1 ~ max_output_seq_len of output_ids_buf_ to output_tensors->at(0).data

    FT_CHECK_WITH_INFO(input_tensors->size() >= 3, "input_tensors->size() >= 3");
    FT_CHECK_WITH_INFO(output_tensors->size() >= 2, "output_tensors->size() >= 2");
    FT_CHECK(input_tensors->at("input_ids").shape.size() == 2);
    FT_CHECK(input_tensors->at("input_lengths").shape.size() == 1);
    FT_CHECK(input_tensors->find("output_seq_len") != input_tensors->end()
             && input_tensors->at("output_seq_len").shape.size() == 1);
    FT_CHECK(output_tensors->at("output_ids").shape.size() == 3);
    FT_CHECK(output_tensors->at("sequence_length").shape.size() == 2);
    FT_CHECK_WITH_INFO(input_tensors->at("input_ids").shape[0] == output_tensors->at("output_ids").shape[0],
                       "input_tensors->at(\"input_ids\").shape[0] == output_tensors->at(\"output_ids\").shape[0]");

    const size_t batch_size = output_tensors->at("output_ids").shape[0];
    const size_t beam_width = output_tensors->at("output_ids").shape[1];

    PromptLearningType request_prompt_type = PromptLearningType::no_prompt;
    int                valid_prompt_inputs = input_tensors->count("request_prompt_type")
                              + input_tensors->count("request_prompt_lengths")
                              + input_tensors->count("request_prompt_embedding");

    if (valid_prompt_inputs == 3) {
        request_prompt_type = static_cast<PromptLearningType>(input_tensors->at("request_prompt_type").getVal<int>());
        FT_LOG_INFO("Apply prompt embedding from input, will ignore task name ids");
    }
    else if (valid_prompt_inputs > 0) {
        FT_LOG_WARNING(
            "Prompts not applied: request_prompt_embedding, request_prompt_lengths, request_prompt_type are all needed!");
    }
    if (request_prompt_type == PromptLearningType::prefix_prompt) {
        FT_LOG_WARNING("Request prompt doesn't support prefix prompt currently!");
    }

    // Prefix Prompt Inputs
    // Padding works as follows: p p x x i i i x x --> p p i i i x x x x (p denotes prompt, i denotes input, x denotes
    // pad)
    // TODO (perkzz): move unnecessary paddings
    const int* prompt_learning_task_name_ids =
        input_tensors->count("prompt_learning_task_name_ids") ?
            input_tensors->at("prompt_learning_task_name_ids").getPtr<const int>() :
            nullptr;
    has_prefix_prompt_ =
        (prompt_learning_task_name_ids != nullptr) && (prompt_learning_type_ == PromptLearningType::prefix_prompt);
    int max_prefix_prompt_length = 0;

    FT_CHECK_WITH_INFO(
        !(prompt_learning_task_name_ids != nullptr
          && (prompt_learning_type_ == PromptLearningType::no_prompt
              || prompt_learning_type_ == PromptLearningType::soft_prompt)),
        "prompt_learning_type is prefix_prompt either p_prompt_tuning when prompt_learning_task_name_ids are provided.");

    // NOTE: Prefix Prompt PreProcessing
    // get prefix_prompt_weight for each batch --> shape [batch, beam_width]
    // --> ptrs with shape [num_layers, 2, num_heads, perfix_seq_len, size_per_head]
    std::vector<const T*> prefix_prompt_weight_batch_ptrs;
    std::vector<int>      prefix_prompt_lengths;
    if (has_prefix_prompt_) {
        for (int bs_id = 0; bs_id < batch_size; ++bs_id) {
            int task_id = prompt_learning_task_name_ids[bs_id];
            // throw errors when prompt task_name_ids are not found
            std::pair<const T*, int> prefix_prompt_weight_length_pair;
            try {
                prefix_prompt_weight_length_pair = gpt_weights->prompt_learning_table.at(task_id);
            }
            catch (const std::out_of_range& oor) {
                FT_LOG_ERROR("prefix_prompt_weights_lengths not found for prompt task id: " + task_id);
                throw oor;
            }
            for (int bw_id = 0; bw_id < beam_width; ++bw_id) {
                prefix_prompt_weight_batch_ptrs.push_back(prefix_prompt_weight_length_pair.first);
                prefix_prompt_lengths.push_back(prefix_prompt_weight_length_pair.second);
            }
        }

        max_prefix_prompt_length = *max_element(prefix_prompt_lengths.begin(), prefix_prompt_lengths.end());

        FT_LOG_DEBUG("max_prefix_prompt_length: %d", max_prefix_prompt_length);

        if (max_prefix_prompt_length == 0) {
            has_prefix_prompt_ = false;
            FT_LOG_DEBUG("prompts are not applied !");
        }
    }

    int max_input_length = input_tensors->at("input_ids").shape[1];
    FT_CHECK_WITH_INFO(!(max_input_length == 0 && max_prefix_prompt_length > 0),
                       "Prefix Prompt should come with inputs!");

    // Prefix Soft Prompt
    has_prefix_soft_prompt_ = request_prompt_type == PromptLearningType::soft_prompt;
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;
    const size_t limit_len_offset   = max_prefix_soft_prompt_length + (max_input_length == 0 ? 1 : 0);
    const size_t max_output_seq_len = input_tensors->at("output_seq_len").max<uint32_t>() + limit_len_offset;
    const size_t max_seq_len        = max_output_seq_len;
    // max cache seq len should include max prefix prompt length as it has k/v states
    const size_t max_cache_seq_len = max_output_seq_len + max_prefix_prompt_length;
    if (max_cache_seq_len < max_seq_len) {
        FT_LOG_WARNING("max_cache_seq_len (%d) is less than max_seq_len (%d). "
                       "Note that this reduces the memory cost of k/v cache, but may hurt the accuracy.",
                       max_cache_seq_len,
                       max_seq_len);
    }
    else if (max_cache_seq_len > max_seq_len) {
        FT_LOG_WARNING("max_cache_seq_len (%d) is larger than max_seq_len (%d). "
                       "This may lead to additional memory cost. Suggest to use smaller max_cache_seq_len.",
                       max_cache_seq_len,
                       max_seq_len);
    }
    const cudaDataType_t gemm_data_type = getCudaDataType<T>();
    allocateBuffer(
        batch_size, beam_width, max_seq_len, max_cache_seq_len, max_input_length + max_prefix_soft_prompt_length);
    setSeqLimitLen(seq_limit_len_, input_tensors->at("output_seq_len"), limit_len_offset, batch_size);

    sync_check_cuda_error();
    {
        TensorMap input_map(*input_tensors);
        dynamic_decode_layer_->setup(batch_size, beam_width, &input_map);
        handleOptArg(&input_map, "start_id", start_ids_buf_, start_id_, batch_size);
        handleOptArg(&input_map, "end_id", end_ids_buf_, end_id_, batch_size);
    }

    const DataType data_type = getTensorType<T>();

    const std::vector<size_t> self_k_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    size_per_head_ / (16 / sizeof(T)),
                                                    max_cache_seq_len,
                                                    16 / sizeof(T)};
    const std::vector<size_t> self_v_cache_shape = {num_layer_ / pipeline_para_.world_size_,
                                                    batch_size * beam_width,
                                                    local_head_num_,
                                                    max_cache_seq_len,
                                                    size_per_head_};

    // initialize the output ids and parent ids
    cudaMemsetAsync(output_ids_buf_, 0, sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    cudaMemsetAsync(parent_ids_buf_, 0, sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    cudaMemsetAsync(masked_tokens_, false, sizeof(bool) * batch_size * beam_width * max_cache_seq_len, stream_);
    cudaMemsetAsync(tiled_total_padding_count_, 0, sizeof(int) * batch_size * beam_width, stream_);
    if (beam_width > 1) {
        cudaMemsetAsync(cache_indirections_[0], 0, 2 * sizeof(int) * batch_size * beam_width * max_seq_len, stream_);
    }

    int  compact_size;
    bool use_shared_contexts = (shared_contexts_ratio_ > 0.0f) && (max_input_length >= 1) && (batch_size > 1);
    if (use_shared_contexts) {
        invokeFindContextDups(shared_contexts_idx_,
                              batch_to_compact_idx_,
                              compact_idx_,
                              compact_size_,
                              input_tensors->at("input_ids").getPtr<int>(),
                              batch_size,
                              beam_width,
                              max_input_length,
                              stream_);
        cudaD2Hcpy(&compact_size, compact_size_, 1);
        use_shared_contexts = compact_size <= shared_contexts_ratio_ * batch_size;
        sync_check_cuda_error();
    }

    // Prefix prompts
    if (has_prefix_prompt_) {
        cudaMemcpyAsync(prompt_learning_weight_batch_,
                        prefix_prompt_weight_batch_ptrs.data(),
                        sizeof(T*) * batch_size * beam_width,
                        cudaMemcpyDefault,
                        stream_);
        cudaMemcpyAsync(tiled_prompt_lengths_buf_,
                        prefix_prompt_lengths.data(),
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyDefault,
                        stream_);
    }

    sync_check_cuda_error();

    // handle first step
    if (has_prefix_prompt_ || has_prefix_soft_prompt_ || max_input_length > 1) {
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("input_ids").getPtr<int>(),
                            input_tensors->at("input_lengths").getPtr<const int>(),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();

        if (has_prefix_soft_prompt_) {
            inputIdsEmbeddingLookupPosEncodingSoftPromptParam<T> param;
            param.from_tensor                   = context_decoder_input_buf_;
            param.output_ids                    = output_ids_buf_;
            param.input_lengths                 = tiled_input_lengths_buf_;
            param.embedding_table               = gpt_weights->pre_decoder_embedding_table;
            param.pos_table                     = gpt_weights->position_encoding_table;
            param.prefix_soft_prompt_embedding  = input_tensors->at("request_prompt_embedding").getPtr<float>();
            param.prefix_soft_prompt_lengths    = input_tensors->at("request_prompt_lengths").getPtr<int>();
            param.input_ids                     = tiled_input_ids_buf_;
            param.start_step                    = 1;
            param.max_input_length              = max_input_length;
            param.max_prefix_soft_prompt_length = max_prefix_soft_prompt_length;
            param.batch_size                    = batch_size;
            param.beam_width                    = beam_width;
            param.hidden_units                  = hidden_units_;
            param.stream                        = stream_;

            invokeInputIdsEmbeddingLookupPosEncodingSoftPrompt(param);
            sync_check_cuda_error();
            max_input_length += max_prefix_soft_prompt_length;  // view soft_prompt as input
        }
        else {
            invokeInputIdsEmbeddingLookupPosEncoding(context_decoder_input_buf_,
                                                     output_ids_buf_,
                                                     gpt_weights->pre_decoder_embedding_table,
                                                     gpt_weights->position_encoding_table,
                                                     pPromptTuningParam<T>{},  // no p/prompt tuning
                                                     tiled_input_ids_buf_,
                                                     1,
                                                     max_input_length,
                                                     max_input_length,
                                                     batch_size * beam_width,
                                                     hidden_units_,
                                                     stream_);
            sync_check_cuda_error();
        }

        invokeBuildDecoderAttentionMask(input_attention_mask_,
                                        tiled_input_lengths_buf_,
                                        tiled_prompt_lengths_buf_,
                                        batch_size * beam_width,
                                        max_input_length,
                                        max_prefix_prompt_length,
                                        stream_);
        sync_check_cuda_error();

        std::unordered_map<std::string, Tensor> decoder_input_tensors{
            {"decoder_input",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                    context_decoder_input_buf_}},
            {"attention_mask",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width,
                     1,
                     (size_t)max_input_length,
                     (size_t)(max_input_length + max_prefix_prompt_length)},
                    input_attention_mask_}},
            {"input_lengths", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, tiled_input_lengths_buf_}},
            {"d_prefix_prompt_batch",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width},
                    has_prefix_prompt_ ? prompt_learning_weight_batch_ : nullptr}},
            {"d_prefix_prompt_lengths",
             Tensor{MEMORY_GPU,
                    TYPE_INT32,
                    {batch_size * beam_width},
                    has_prefix_prompt_ ? tiled_prompt_lengths_buf_ : nullptr}}};

        if (use_shared_contexts) {
            decoder_input_tensors.insert(
                {"compact_idx", Tensor(MEMORY_GPU, TYPE_INT32, {(size_t)compact_size}, compact_idx_)});
            decoder_input_tensors.insert(
                {"batch_to_compact_idx",
                 Tensor(MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, batch_to_compact_idx_)});
        }

        std::unordered_map<std::string, Tensor> decoder_output_tensors{
            {"decoder_output",
             Tensor{MEMORY_GPU,
                    data_type,
                    {batch_size * beam_width, (size_t)max_input_length, hidden_units_},
                    context_decoder_output_buf_}},
            {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
            {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}},
            {"last_token_hidden_units",
             Tensor{MEMORY_GPU, data_type, {batch_size * beam_width, hidden_units_}, decoder_output_buf_}}};

        gpt_context_decoder_->forward(
            &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
        sync_check_cuda_error();
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths_,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        sync_check_cuda_error();
    }
    else if (max_input_length == 0) {
        FT_CHECK(prompt_learning_type_ == PromptLearningType::no_prompt
                 && request_prompt_type == PromptLearningType::no_prompt);  // Not support prompts in this case
        max_input_length++;
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths_,
                                 output_ids_buf_,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        std::vector<int> h_input_lengths(batch_size * beam_width, 1);
        cudaMemcpyAsync(tiled_input_lengths_buf_,
                        h_input_lengths.data(),
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyHostToDevice,
                        stream_);
        sync_check_cuda_error();
    }
    else if (max_input_length == 1) {
        FT_CHECK(prompt_learning_type_ == PromptLearningType::no_prompt
                 && request_prompt_type == PromptLearningType::no_prompt);  // Not support prompts in this case
        invokeDecodingInitialize(finished_buf_,
                                 sequence_lengths_,
                                 nullptr,
                                 cum_log_probs_,
                                 start_ids_buf_,
                                 batch_size,
                                 beam_width,
                                 max_input_length - 1,
                                 stream_);
        sync_check_cuda_error();
        invokeTileGptInputs(tiled_input_ids_buf_,
                            tiled_input_lengths_buf_,
                            input_tensors->at("input_ids").getPtr<int>(),
                            input_tensors->at("input_lengths").getPtr<const int>(),
                            batch_size,
                            beam_width,
                            max_input_length,
                            stream_);
        sync_check_cuda_error();

        cudaMemcpyAsync(output_ids_buf_,
                        tiled_input_ids_buf_,
                        sizeof(int) * batch_size * beam_width,
                        cudaMemcpyDeviceToDevice,
                        stream_);
    }

    if (vocab_size_ == vocab_size_padded_) {
        padded_embedding_kernel_ptr_ = gpt_weights->post_decoder_embedding.kernel;
    }
    else {
        cudaMemcpyAsync(padded_embedding_kernel_,
                        gpt_weights->post_decoder_embedding.kernel,
                        sizeof(T) * vocab_size_ * hidden_units_,
                        cudaMemcpyDeviceToDevice,
                        stream_);
        sync_check_cuda_error();
    }

    invokeMaskPaddingTokens(masked_tokens_,
                            input_tensors->at("input_lengths").getPtr<const int>(),  // not_tiled
                            tiled_prompt_lengths_buf_,
                            max_cache_seq_len,
                            max_input_length + max_prefix_prompt_length,
                            0,
                            batch_size,
                            beam_width,
                            stream_);

    for (int step = max_input_length; step < (int)max_output_seq_len; step++) {
        const int src_indir_idx = (step - max_input_length) % 2;
        const int tgt_indir_idx = 1 - src_indir_idx;

        const size_t local_batch_size = getLocalBatchSize(batch_size, 1, pipeline_para_.world_size_);
        FT_CHECK(batch_size % local_batch_size == 0);
        const size_t iteration_num = batch_size / local_batch_size;
        *generation_should_stop_   = true;

        for (uint ite = 0; ite < iteration_num; ++ite) {
            const int id_offset               = ite * local_batch_size * beam_width;
            const int hidden_units_offset     = id_offset * hidden_units_;
            const int vocab_size_units_offset = id_offset * vocab_size_padded_;

            if (!(max_input_length > 1 && step == max_input_length)) {
                if (pipeline_para_.rank_ == 0) {
                    invokeEmbeddingLookupPosEncodingPadCount(decoder_input_buf_ + hidden_units_offset,
                                                             gpt_weights->pre_decoder_embedding_table,
                                                             gpt_weights->position_encoding_table,
                                                             output_ids_buf_ + id_offset,
                                                             tiled_total_padding_count_ + id_offset,
                                                             local_batch_size * beam_width,
                                                             hidden_units_,
                                                             (T)(1.0f),
                                                             step - 1,
                                                             batch_size * beam_width,
                                                             0,
                                                             stream_);
                    sync_check_cuda_error();
                }
                std::unordered_map<std::string, Tensor> decoder_input_tensors{
                    {"decoder_input",
                     Tensor{MEMORY_GPU,
                            data_type,
                            {local_batch_size * beam_width, hidden_units_},
                            decoder_input_buf_ + hidden_units_offset}},
                    {"finished",
                     Tensor{MEMORY_GPU, TYPE_BOOL, {local_batch_size * beam_width}, finished_buf_ + id_offset}},
                    {"sequence_lengths",
                     Tensor{MEMORY_GPU, TYPE_INT32, {local_batch_size * beam_width}, sequence_lengths_ + id_offset}},
                    {"total_padding_tokens",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size * beam_width},
                            tiled_total_padding_count_ + id_offset}},
                    {"d_prefix_prompt_lengths",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size},
                            has_prefix_prompt_ ? (tiled_prompt_lengths_buf_ + id_offset) : nullptr}},
                    {"max_prefix_prompt_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_prefix_prompt_length}},
                    {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
                    {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                    {"ite", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &ite}},
                    {"cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, max_output_seq_len},
                            beam_width > 1 ? cache_indirections_[src_indir_idx] + id_offset * max_output_seq_len :
                                             nullptr}},
                    {"masked_tokens",
                     Tensor{MEMORY_GPU,
                            TYPE_BOOL,
                            {local_batch_size * beam_width, max_cache_seq_len},
                            masked_tokens_ + id_offset * max_cache_seq_len}}};
                std::unordered_map<std::string, Tensor> decoder_output_tensors{
                    {"decoder_output",
                     Tensor{MEMORY_GPU,
                            data_type,
                            {local_batch_size * beam_width, hidden_units_},
                            decoder_output_buf_ + hidden_units_offset}},
                    {"key_cache", Tensor{MEMORY_GPU, data_type, self_k_cache_shape, key_cache_}},
                    {"value_cache", Tensor{MEMORY_GPU, data_type, self_v_cache_shape, value_cache_}}};
                gpt_decoder_->forward(
                    &decoder_output_tensors, &decoder_input_tensors, &gpt_weights->decoder_layer_weights);
            }

            if (pipeline_para_.rank_ == pipeline_para_.world_size_ - 1) {
                invokeLlamaRMSNorm(normed_decoder_output_buf_ + hidden_units_offset,
                                   decoder_output_buf_ + hidden_units_offset,
                                   gpt_weights->post_decoder_layernorm.gamma,
                                   (const T*)nullptr,
                                   layernorm_eps_,
                                   local_batch_size * beam_width,
                                   hidden_units_,
                                   stream_);
                sync_check_cuda_error();

                if (tensor_para_.world_size_ == 1) {
                    float alpha = 1.0f;
                    float beta  = 0.0f;
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          vocab_size_padded_,  // n
                                          local_batch_size * beam_width,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_,
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          normed_decoder_output_buf_ + hidden_units_offset,
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          &beta,
                                          logits_buf_ + vocab_size_units_offset,
                                          CUDA_R_32F,
                                          vocab_size_padded_, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));
                }
                else {
                    FT_CHECK(vocab_size_padded_ % tensor_para_.world_size_ == 0);
                    const int local_vocab_size = vocab_size_padded_ / tensor_para_.world_size_;
                    float     alpha            = 1.0f;
                    float     beta             = 0.0f;
                    cublas_wrapper_->Gemm(CUBLAS_OP_T,
                                          CUBLAS_OP_N,
                                          local_vocab_size,  // n
                                          local_batch_size * beam_width,
                                          hidden_units_,  // k
                                          &alpha,
                                          padded_embedding_kernel_ptr_
                                              + tensor_para_.rank_ * local_vocab_size * hidden_units_,
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          normed_decoder_output_buf_ + hidden_units_offset,
                                          gemm_data_type,
                                          hidden_units_,  // k
                                          &beta,
                                          nccl_logits_buf_ + vocab_size_units_offset
                                              + tensor_para_.rank_ * local_batch_size * beam_width * local_vocab_size,
                                          CUDA_R_32F,
                                          local_vocab_size, /* n */
                                          CUDA_R_32F,
                                          cublasGemmAlgo_t(-1));

                    ftNcclAllGather(nccl_logits_buf_ + vocab_size_units_offset,
                                    nccl_logits_buf_ + vocab_size_units_offset,
                                    local_batch_size * beam_width * local_vocab_size,
                                    tensor_para_.rank_,
                                    tensor_para_,
                                    stream_);
                    invokeTransposeAxis01(logits_buf_ + vocab_size_units_offset,
                                          nccl_logits_buf_ + vocab_size_units_offset,
                                          tensor_para_.world_size_,
                                          local_batch_size * beam_width,
                                          local_vocab_size,
                                          stream_);
                }

                int                                     tmp_local_batch_size       = local_batch_size;
                bool                                    is_initialize_random_table = step == max_input_length;
                std::unordered_map<std::string, Tensor> dynamic_decode_input_tensors{
                    {"logits",
                     Tensor{MEMORY_GPU, TYPE_FP32, {batch_size, beam_width, vocab_size_padded_}, logits_buf_}},
                    // {"embedding_bias", Tensor{MEMORY_GPU, data_type, {vocab_size_padded_}, nullptr}},
                    {"step", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &step}},
                    {"max_input_length", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &max_input_length}},
                    {"input_lengths",
                     Tensor{MEMORY_GPU, TYPE_INT32, {batch_size, beam_width}, tiled_input_lengths_buf_}},
                    {"sequence_limit_length", Tensor{MEMORY_GPU, TYPE_UINT32, {batch_size}, seq_limit_len_}},
                    {"ite", Tensor{MEMORY_CPU, TYPE_UINT32, {1}, &ite}},
                    {"src_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, max_output_seq_len},
                            cache_indirections_[src_indir_idx] + id_offset * max_output_seq_len}},
                    {"local_batch_size", Tensor{MEMORY_CPU, TYPE_INT32, {1}, &tmp_local_batch_size}},
                    {"end_id", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size}, end_ids_buf_}},
                    {"is_initialize_random_table", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &is_initialize_random_table}}};

                for (auto t = input_tensors->begin(); t != input_tensors->end(); ++t) {
                    if (dynamic_decode_input_tensors.find(t->first) == dynamic_decode_input_tensors.end()) {
                        dynamic_decode_input_tensors.insert(*t);
                    }
                }

                // common outputs
                bool                                    subbatch_should_stop = false;
                std::unordered_map<std::string, Tensor> dynamic_decode_output_tensors{
                    {"output_ids",
                     Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, output_ids_buf_}},
                    {"finished", Tensor{MEMORY_GPU, TYPE_BOOL, {batch_size * beam_width}, finished_buf_}},
                    // cum_log_probs is necessary for beam search, while it is optional for sampling.
                    {"cum_log_probs",
                     Tensor{MEMORY_GPU,
                            TYPE_FP32,
                            {batch_size * beam_width},
                            ((beam_width > 1) || (output_tensors->count("cum_log_probs") > 0)) ? cum_log_probs_ :
                                                                                                 nullptr}},
                    {"output_log_probs",
                     Tensor{MEMORY_GPU,
                            TYPE_FP32,
                            {max_seq_len, batch_size, beam_width},
                            output_tensors->count("output_log_probs") > 0
                                    && output_tensors->at("output_log_probs").data != nullptr ?
                                output_log_probs_buf_ :
                                nullptr}},
                    {"parent_ids",
                     Tensor{MEMORY_GPU, TYPE_INT32, {max_seq_len, batch_size, beam_width}, parent_ids_buf_}},
                    {"sequence_length", Tensor{MEMORY_GPU, TYPE_INT32, {batch_size * beam_width}, sequence_lengths_}},
                    {"tgt_cache_indirection",
                     Tensor{MEMORY_GPU,
                            TYPE_INT32,
                            {local_batch_size, beam_width, max_output_seq_len},
                            cache_indirections_[tgt_indir_idx] + id_offset * max_output_seq_len}},
                    {"should_stop", Tensor{MEMORY_CPU, TYPE_BOOL, {1}, &subbatch_should_stop}}};

                for (auto t = output_tensors->begin(); t != output_tensors->end(); ++t) {
                    // Handle exceptions.
                    if (t->first == "cum_log_probs" || t->first == "output_log_probs") {
                        continue;
                    }
                    dynamic_decode_output_tensors.insert(*t);
                }

                dynamic_decode_layer_->forward(&dynamic_decode_output_tensors, &dynamic_decode_input_tensors);
                *generation_should_stop_ &= subbatch_should_stop;
            }
        }

        if (pipeline_para_.world_size_ > 1) {
            ftNcclGroupStart();
            ftNcclBroadCast(output_ids_buf_ + step * batch_size * beam_width,
                            batch_size * beam_width,
                            pipeline_para_.world_size_ - 1,
                            pipeline_para_,
                            stream_);

            ftNcclBroadCast(
                sequence_lengths_, batch_size * beam_width, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);

            ftNcclBroadCast(generation_should_stop_, 1, pipeline_para_.world_size_ - 1, pipeline_para_, stream_);

            if (beam_width > 1) {
                ftNcclBroadCast(cache_indirections_[tgt_indir_idx],
                                batch_size * beam_width * max_output_seq_len,
                                pipeline_para_.world_size_ - 1,
                                pipeline_para_,
                                stream_);
            }
            ftNcclGroupEnd();
            // throw errors when detected
            ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
            sync_check_cuda_error();
        }

        if (*generation_should_stop_) {
            break;
        }
        if (token_generated_cb_ && (step + 1) % token_generated_cb_step_ == 0 && step + 1 < (int)max_output_seq_len) {
            setOutputTensors(output_tensors, input_tensors, max_input_length, max_output_seq_len);
            sendTensorsToFirstPipelineNode(output_tensors, input_tensors);

            if (pipeline_para_.rank_ == 0 && tensor_para_.rank_ == 0) {
                token_generated_cb_(output_tensors, token_generated_ctx_);
            }
        }
        if (step == max_input_length) {
            /* We have just finished processing input: update the padding count:
             * total_padding_count += (max_input_length - input_lengths)
             * if has prefix prompts, += (max_prefix_prompt_length - prompt_length)
             */
            invokeUpdatePaddingCount(tiled_total_padding_count_,
                                     input_tensors->at("input_lengths").getPtr<const int>(),  // not_tiled
                                     has_prefix_prompt_ ? tiled_prompt_lengths_buf_ : (const int*)nullptr,
                                     max_input_length,
                                     has_prefix_prompt_ ? max_prefix_prompt_length : 0,
                                     batch_size,
                                     beam_width,
                                     stream_);
        }
    }

    setOutputTensors(output_tensors, input_tensors, max_input_length, max_output_seq_len);
    sendTensorsToFirstPipelineNode(output_tensors, input_tensors);
}

template<typename T>
void Llama<T>::sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                              const std::unordered_map<std::string, Tensor>* input_tensors)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (pipeline_para_.world_size_ == 1) {
        // throw errors when detected
        ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
        return;
    }

    const auto pp_rank = pipeline_para_.rank_;

    ftNcclGroupStart();
    for (auto const& it : *output_tensors) {
        if (it.second.data == nullptr) {
            continue;
        }

        if (pp_rank == pipeline_para_.world_size_ - 1) {
            ftNcclSend(it.second.getPtr<char>(), it.second.sizeBytes(), 0, pipeline_para_, stream_);
        }
        else if (pp_rank == 0) {
            ftNcclRecv(it.second.getPtr<char>(),
                       it.second.sizeBytes(),
                       pipeline_para_.world_size_ - 1,
                       pipeline_para_,
                       stream_);
        }
    }
    ftNcclGroupEnd();
    // throw errors when detected
    ftNcclStreamSynchronize(tensor_para_, pipeline_para_, stream_);
}

template<typename T>
void Llama<T>::setOutputTensors(std::unordered_map<std::string, Tensor>*       output_tensors,
                                const std::unordered_map<std::string, Tensor>* input_tensors,
                                const size_t                                   max_input_length,
                                const size_t                                   max_output_seq_len)
{
    FT_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (pipeline_para_.rank_ != pipeline_para_.world_size_ - 1) {
        return;
    }

    const size_t batch_size       = output_tensors->at("output_ids").shape[0];
    const size_t beam_width       = output_tensors->at("output_ids").shape[1];
    int*         sequence_lengths = output_tensors->at("sequence_length").getPtr<int>();
    const size_t max_prefix_soft_prompt_length =
        has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_embedding").shape[1] : 0;

    cudaAutoCpy(sequence_lengths, sequence_lengths_, output_tensors->at("sequence_length").size(), stream_);
    if (input_tensors->at("input_ids").shape[1] == 0) {
        // TODO: D2D sequence_lenghts
        if (beam_width > 1) {
            // For beam search, do gather_tree
            // take output_parent_ids as inter buffer
            invokeGatherTree(transposed_output_ids_buf_,
                             sequence_lengths,
                             max_output_seq_len,
                             batch_size,
                             beam_width,
                             output_ids_buf_ + batch_size * beam_width,
                             parent_ids_buf_ + batch_size * beam_width,
                             end_ids_buf_,
                             stream_);

            // transpose and take output_parent_ids as inter buffer
            invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                                  transposed_output_ids_buf_,
                                  max_output_seq_len - 1,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
        else {
            // For sampling, only copy the results to output_tensor
            invokeTransposeAxis01(output_tensors->at("output_ids").getPtr<int>(),
                                  output_ids_buf_ + batch_size * beam_width,
                                  max_output_seq_len - 1,
                                  batch_size * beam_width,
                                  1,
                                  stream_);
        }
    }
    else {

        // For sampling, it is equivalent to all parent ids are 0.
        gatherTreeParam param;
        param.beams                = transposed_output_ids_buf_;
        param.max_sequence_lengths = sequence_lengths;
        // add sequence_length 1 here because the sequence_length of time step t is t - 1
        param.max_sequence_length_final_step = 1;
        param.max_time                       = max_output_seq_len;
        param.batch_size                     = batch_size;
        param.beam_width                     = beam_width;
        param.step_ids                       = output_ids_buf_;
        param.parent_ids                     = beam_width == 1 ? nullptr : parent_ids_buf_;
        param.end_tokens                     = end_ids_buf_;
        param.max_input_length               = max_input_length;
        param.prefix_soft_prompt_lengths =
            has_prefix_soft_prompt_ ? input_tensors->at("request_prompt_lengths").getPtr<int>() : nullptr;
        param.input_lengths                   = tiled_input_lengths_buf_;
        param.max_prefix_soft_prompt_length   = max_prefix_soft_prompt_length;
        param.max_input_without_prompt_length = max_input_length;
        param.stream                          = stream_;
        param.output_ids                      = output_tensors->at("output_ids").getPtr<int>();
        invokeGatherTree(param);
        sync_check_cuda_error();
    }
    if ((output_tensors->count("output_log_probs") > 0 && output_tensors->at("output_log_probs").data != nullptr)) {
        invokeTransposeAxis01(output_tensors->at("output_log_probs").getPtr<float>(),
                              output_log_probs_buf_,
                              input_tensors->at("output_seq_len").max<uint32_t>() - max_input_length,
                              batch_size * beam_width,
                              1,
                              stream_);
    }
    // Return the cumulative log probability if requested.
    if (output_tensors->count("cum_log_probs") > 0) {
        Tensor cum_log_probs = output_tensors->at("cum_log_probs");
        FT_CHECK_WITH_INFO(cum_log_probs.size() == batch_size * beam_width,
                           "The shape of cum_log_probs does not match with batch_size x beam_width.");
        cudaAutoCpy(cum_log_probs.getPtr<float>(), cum_log_probs_, cum_log_probs.size(), stream_);
    }
}

template<typename T>
size_t Llama<T>::getPipelineParallelRank()
{
    return pipeline_para_.rank_;
}

template<typename T>
size_t Llama<T>::getPipelineParallelSize()
{
    return pipeline_para_.world_size_;
}

template<typename T>
size_t Llama<T>::getTensorParallelRank()
{
    return tensor_para_.rank_;
}

template<typename T>
size_t Llama<T>::getTensorParallelSize()
{
    return tensor_para_.world_size_;
}

template<typename T>
bool* Llama<T>::getFinishBuffer()
{
    return finished_buf_;
}

template class Llama<float>;
template class Llama<half>;
#ifdef ENABLE_BF16
template class Llama<__nv_bfloat16>;
#endif
}  // namespace fastertransformer
