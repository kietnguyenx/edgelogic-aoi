#include "aoi_engine.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <iostream>

namespace edgelogic {

bool AOIEngine::load(const EngineConfig& config) {
    config_ = config;
    try {
        auto model = core_.read_model(config_.model_xml_path);
        ov::AnyMap props;
        if (config_.performance_hint == "THROUGHPUT") {
            props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
        } else {
            props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
        }
        compiled_model_ = core_.compile_model(model, config_.device, props);
        sync_request_ = compiled_model_.create_infer_request();
        loaded_ = true;
        return true;
    } catch (const std::exception& ex) {
        std::cerr << "[AOIEngine] Failed to load model: " << ex.what() << "\n";
        loaded_ = false;
        return false;
    }
}

int AOIEngine::optimal_num_requests() const {
    if (!loaded_) return 1;
    try {
        return static_cast<int>(compiled_model_.get_property(ov::optimal_number_of_infer_requests));
    } catch (...) {
        return 4;
    }
}

void AOIEngine::preprocess_to_buffer(const cv::Mat& frame_bgr, float* dst_nchw) const {
    const int sz = config_.image_size;
    cv::Mat resized, rgb, f32;
    cv::resize(frame_bgr, resized, cv::Size(sz, sz), 0, 0, cv::INTER_AREA);
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(f32, CV_32FC3, 1.0 / 255.0);

    const size_t plane = static_cast<size_t>(sz) * static_cast<size_t>(sz);
    std::vector<cv::Mat> channels = {
        cv::Mat(sz, sz, CV_32FC1, dst_nchw),
        cv::Mat(sz, sz, CV_32FC1, dst_nchw + plane),
        cv::Mat(sz, sz, CV_32FC1, dst_nchw + 2 * plane)
    };
    cv::split(f32, channels);
}

InferenceResult AOIEngine::infer_single(const cv::Mat& frame_bgr) {
    InferenceResult out;
    if (!loaded_ || frame_bgr.empty()) {
        return out;
    }

    ov::Tensor input_tensor = sync_request_.get_input_tensor(0);
    preprocess_to_buffer(frame_bgr, input_tensor.data<float>());

    auto t0 = std::chrono::steady_clock::now();
    sync_request_.infer();
    auto t1 = std::chrono::steady_clock::now();
    out.infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    ov::Tensor score_tensor = sync_request_.get_output_tensor(0);
    out.anomaly_score = static_cast<double>(score_tensor.data<const float>()[0]);

    if (compiled_model_.outputs().size() > 1) {
        ov::Tensor map_tensor = sync_request_.get_output_tensor(1);
        const auto shape = map_tensor.get_shape();
        if (shape.size() >= 2) {
            int h = static_cast<int>(shape[shape.size() - 2]);
            int w = static_cast<int>(shape[shape.size() - 1]);
            cv::Mat raw_map(h, w, CV_32FC1, const_cast<float*>(map_tensor.data<const float>()));
            out.anomaly_map = raw_map.clone();
        }
    }
    return out;
}

LatencyStats AOIEngine::benchmark_async(const std::vector<cv::Mat>& frames, int num_streams) {
    MetricsTimer timer;
    if (!loaded_ || frames.empty()) {
        return timer.summarize();
    }

    int n_req = (num_streams > 0) ? num_streams : std::max(1, optimal_num_requests());
    std::vector<ov::InferRequest> pool;
    pool.reserve(static_cast<size_t>(n_req));
    for (int i = 0; i < n_req; ++i) {
        pool.push_back(compiled_model_.create_infer_request());
    }

    // Warmup all requests once
    for (int i = 0; i < n_req; ++i) {
        preprocess_to_buffer(frames[i % frames.size()], pool[i].get_input_tensor(0).data<float>());
        pool[i].start_async();
    }
    for (int i = 0; i < n_req; ++i) {
        pool[i].wait();
    }

    // Timed pipelined execution
    std::vector<std::chrono::steady_clock::time_point> start_times(static_cast<size_t>(n_req));
    std::vector<bool> active(static_cast<size_t>(n_req), false);

    auto wall_start = std::chrono::steady_clock::now();
    for (size_t idx = 0; idx < frames.size(); ++idx) {
        size_t slot = idx % static_cast<size_t>(n_req);
        if (active[slot]) {
            pool[slot].wait();
            auto now = std::chrono::steady_clock::now();
            timer.record_ms(std::chrono::duration<double, std::milli>(now - start_times[slot]).count());
        }
        preprocess_to_buffer(frames[idx], pool[slot].get_input_tensor(0).data<float>());
        start_times[slot] = std::chrono::steady_clock::now();
        pool[slot].start_async();
        active[slot] = true;
    }

    for (size_t slot = 0; slot < static_cast<size_t>(n_req); ++slot) {
        if (active[slot]) {
            pool[slot].wait();
            auto now = std::chrono::steady_clock::now();
            timer.record_ms(std::chrono::duration<double, std::milli>(now - start_times[slot]).count());
        }
    }
    auto wall_end = std::chrono::steady_clock::now();
    double total_wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    return timer.summarize(total_wall_ms);
}

}  // namespace edgelogic