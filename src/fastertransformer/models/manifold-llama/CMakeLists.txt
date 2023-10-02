# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.8)

add_library(MnfdLlamaDecoderLayerWeight STATIC LlamaDecoderLayerWeight.cc)
set_property(TARGET MnfdLlamaDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET MnfdLlamaDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(MnfdLlamaDecoderLayerWeight PUBLIC memory_utils cuda_utils logger)

add_library(MnfdLlamaDecoder STATIC LlamaDecoder.cc)
set_property(TARGET MnfdLlamaDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET MnfdLlamaDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(MnfdLlamaDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelDecoderSelfAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      MnfdLlamaDecoderLayerWeight
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(MnfdLlamaContextDecoder STATIC LlamaContextDecoder.cc)
set_property(TARGET MnfdLlamaContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET MnfdLlamaContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(MnfdLlamaContextDecoder PUBLIC -lcudart cublasMMWrapper
                      TensorParallelGptContextAttentionLayer
                      TensorParallelSiluFfnLayer
                      layernorm_kernels
                      add_residual_kernels
                      gpt_kernels
                      tensor
                      nccl_utils
                      cuda_utils
                      logger)

add_library(MnfdLlamaWeight STATIC LlamaWeight.cc)
set_property(TARGET MnfdLlamaWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET MnfdLlamaWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(MnfdLlamaWeight PUBLIC MnfdLlamaDecoderLayerWeight cuda_utils logger)

add_library(MnfdLlama STATIC Llama.cc)
set_property(TARGET MnfdLlama PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET MnfdLlama PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(MnfdLlama PUBLIC -lcudart
                      MnfdLlamaDecoder
                      MnfdLlamaContextDecoder
                      decoding_kernels
                      gpt_kernels
                      DynamicDecodeLayer
                      BaseBeamSearchLayer
                      bert_preprocess_kernels
                      tensor
                      MnfdLlamaWeight
                      cuda_utils
                      logger)
