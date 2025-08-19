#include "navigation_onxx_runtime_prediction_model.hpp"

ONNXInference::ONNXInference(const std::string& model_path,
                             size_t seq_len,
                             size_t feature_dim,
                             size_t num_layers)
    : env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime"),
      session_options(),
      session(env, model_path.c_str(), session_options),
      memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)),
      seq_len(seq_len),
      feature_dim(feature_dim),
      num_layers(num_layers)
{
    std::cout << "Initialized ONNX Runtime session with model: " << model_path << std::endl;

    mean = {0.200176164f, 0.00314905686f, 0.0362309597f, 0.328111807f, 0.013561388f, 0.0000321285141f, 0.00000253172694f};
    scale = {4.13309841f, 3.03982038f, 0.0497949554f, 0.193818527f, 0.0136450762f, 0.0444112188f, 0.000820104808f};

    // Hidden state buffer
    hidden_state_flat.resize(num_layers * 1 * seq_len, 0.0f);
    hidden_state_flat_ptr = hidden_state_flat.data();
    hidden_state_flat_size = hidden_state_flat.size();

    size_t num_input_nodes = session.GetInputCount();
    for (size_t i = 0; i < num_input_nodes; i++)
    {
        input_name_ptrs.push_back(session.GetInputNameAllocated(i, allocator));
        const char* input_name = input_name_ptrs.back().get();
        input_node_names.push_back(input_name);

        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        input_node_dims.push_back(tensor_info.GetShape());
    }

    // Get output info
    size_t num_output_nodes = session.GetOutputCount();
    for (size_t i = 0; i < num_output_nodes; i++)
    {
        output_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
        const char* output_name = output_name_ptrs.back().get();
        output_node_names.push_back(output_name);

        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    }

    // Initialize current values
    m_vx = m_vy = m_omega = m_throttle_fb = m_steering_fb = m_throttle_cmd = m_steering_cmd = 0.0001f;

    // Initialize history buffers
    m_vx_history.resize(seq_len, 0.0001f);
    m_vy_history.resize(seq_len, 0.0001f);
    m_omega_history.resize(seq_len, 0.0001f);
    m_throttle_fb_history.resize(seq_len, 0.0001f);
    m_steering_fb_history.resize(seq_len, 0.0001f);
    m_throttle_cmd_history.resize(seq_len, 0.0001f);
    m_steering_cmd_history.resize(seq_len, 0.0001f);
}

void ONNXInference::set_inputs(float vx, float vy, float omega,
                               float throttle_fb, float steering_fb,
                               float throttle_cmd, float steering_cmd)
{
    auto shift_and_append = [this](std::vector<float>& history, float new_val) {
        for (size_t i = 0; i < seq_len - 1; i++)
            history[i] = history[i + 1];
        history[seq_len - 1] = new_val;
    };

    shift_and_append(m_vx_history, vx);
    shift_and_append(m_vy_history, vy);
    shift_and_append(m_omega_history, omega);
    shift_and_append(m_throttle_fb_history, throttle_fb);
    shift_and_append(m_steering_fb_history, steering_fb);
    shift_and_append(m_throttle_cmd_history, throttle_cmd);
    shift_and_append(m_steering_cmd_history, steering_cmd);

    m_vx = vx;
    m_vy = vy;
    m_omega = omega;
    m_throttle_fb = throttle_fb;
    m_steering_fb = steering_fb;
    m_throttle_cmd = throttle_cmd;
    m_steering_cmd = steering_cmd;
}

std::vector<float> ONNXInference::run_inference(int iterations)
{
    std::vector<float> first_output;

    for (int iter = 0; iter < iterations; iter++)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        // Input sequences using history
        std::vector<std::vector<float>> input_seq(seq_len, std::vector<float>(feature_dim, 0.0f));
        for (size_t t = 0; t < seq_len; ++t)
        {
            input_seq[t][0] = m_vx_history[t];
            input_seq[t][1] = m_vy_history[t];
            input_seq[t][2] = m_omega_history[t];
            input_seq[t][3] = m_throttle_fb_history[t];
            input_seq[t][4] = m_steering_fb_history[t];
            input_seq[t][5] = m_throttle_cmd_history[t];
            input_seq[t][6] = m_steering_cmd_history[t];
        }

        // Flatten input
        std::vector<float> input_seq_flat(seq_len * feature_dim);
        for (size_t t = 0; t < seq_len; ++t)
            for (size_t f = 0; f < feature_dim; ++f)
                input_seq_flat[t * feature_dim + f] = input_seq[t][f];

        // Normalized input
        std::vector<float> norm_input_seq_flat(seq_len * feature_dim);
        for (size_t t = 0; t < seq_len; ++t)
            for (size_t f = 0; f < feature_dim; ++f)
                norm_input_seq_flat[t * feature_dim + f] = (input_seq[t][f] - mean[f]) / scale[f];

        // Create tensors
        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, input_seq_flat.data(), input_seq_flat.size(),
            input_node_dims[0].data(), input_node_dims[0].size()));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, norm_input_seq_flat.data(), norm_input_seq_flat.size(),
            input_node_dims[1].data(), input_node_dims[1].size()));

        input_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info, hidden_state_flat_ptr, hidden_state_flat_size,
            input_node_dims[2].data(), input_node_dims[2].size()));

        // Run inference
        std::vector<Ort::Value> output_tensors = session.Run(
            Ort::RunOptions{nullptr},
            input_node_names.data(), input_tensors.data(), input_node_names.size(),
            output_node_names.data(), output_node_names.size());

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        std::cout << "Iteration " << iter << " inference time: " << duration.count() << " microseconds" << std::endl;

        // Process outputs
        if (!output_tensors.empty() && output_tensors[0].IsTensor())
        {
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
            size_t num_elements = tensor_info.GetElementCount();

            // Save first output to return
            first_output.assign(output_data, output_data + num_elements);
        }

        // Update hidden state if present
        for (size_t i = 0; i < output_tensors.size(); i++)
        {
            const char* output_name = output_node_names[i];
            if (std::string(output_name) == "hidden_state" && output_tensors[i].IsTensor())
            {
                hidden_state_flat_ptr = output_tensors[i].GetTensorMutableData<float>();
            }
        }
    }

    return first_output;
}

void ONNXInference::print_shape(const std::vector<int64_t>& shape)
{
    std::cout << "Shape: [";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        std::cout << shape[i];
        if (i < shape.size() - 1)
            std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void ONNXInference::normalize_inputs(const std::vector<std::vector<float>>& inputs,
                                     const std::vector<float>& mean,
                                     const std::vector<float>& scale,
                                     std::vector<std::vector<float>>& norm_inputs)
{
    size_t seq_len = inputs.size();
    size_t feature_dim = mean.size();
    norm_inputs.resize(seq_len, std::vector<float>(feature_dim, 0.0f));

    for (size_t t = 0; t < seq_len; ++t)
        for (size_t f = 0; f < feature_dim; ++f)
            norm_inputs[t][f] = (inputs[t][f] - mean[f]) / scale[f];
}
