#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <onnxruntime_cxx_api.h>

class ONNXInference
{
public:
    ONNXInference(const std::string& model_path, size_t seq_len, size_t feature_dim, size_t num_layers);
    ~ONNXInference() = default;

    std::vector<float> run_inference(int iterations = 1);

    void set_inputs(float vx, float vy, float omega,
                               float throttle_fb, float steering_fb,
                               float throttle_cmd, float steering_cmd);

private:
    // Utility functions
    void print_shape(const std::vector<int64_t>& shape);
    void normalize_inputs(const std::vector<std::vector<float>>& inputs,
                          const std::vector<float>& mean,
                          const std::vector<float>& scale,
                          std::vector<std::vector<float>>& norm_inputs);
                          

private:
    // ONNX Runtime variables
    Ort::Env env;
    Ort::SessionOptions session_options;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info;

    // Model input/output info
    std::vector<Ort::AllocatedStringPtr> input_name_ptrs;
    std::vector<const char*> input_node_names;
    std::vector<std::vector<int64_t>> input_node_dims;
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    std::vector<const char*> output_node_names;

    // Input parameters
    size_t seq_len;
    size_t feature_dim;
    size_t num_layers;

    // Data
    std::vector<float> mean;
    std::vector<float> scale;
    std::vector<float> hidden_state_flat;
    float* hidden_state_flat_ptr;
    size_t hidden_state_flat_size;

    float m_vx, m_vy, m_omega, m_throttle_fb, m_steering_fb, m_throttle_cmd, m_steering_cmd;
    
    std::vector<float> m_vx_history;
    std::vector<float> m_vy_history;
    std::vector<float> m_omega_history;
    std::vector<float> m_throttle_fb_history;
    std::vector<float> m_steering_fb_history;
    std::vector<float> m_throttle_cmd_history;
    std::vector<float> m_steering_cmd_history;
};