#pragma once

#include "metrics_timer.hpp"

#include <opencv2/core.hpp>
#include <openvino/openvino.hpp>

#include <string>
#include <vector>

namespace edgelogic {

struct EngineConfig {
    std::string model_xml_path;
    std::string device = "CPU";
    std::string performance_hint = "LATENCY";  // "LATENCY" or "THROUGHPUT"
    int image_size = 256;
};

struct InferenceResult {
    double anomaly_score = 0.0;
    double infer_ms = 0.0;
    cv::Mat anomaly_map;
};

class AOIEngine {
public:
    AOIEngine() = default;

    bool load(const EngineConfig& config);
    InferenceResult infer_single(const cv::Mat& frame_bgr);
    LatencyStats benchmark_async(const std::vector<cv::Mat>& frames, int num_streams = 0);

    std::string device_name() const { return config_.device; }
    int optimal_num_requests() const;

private:
    void preprocess_to_buffer(const cv::Mat& frame_bgr, float* dst_nchw) const;

    EngineConfig config_;
    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest sync_request_;
    bool loaded_ = false;
};

}  // namespace edgelogic

