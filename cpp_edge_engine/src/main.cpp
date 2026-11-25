#include "aoi_engine.hpp"
#include "conformal_triage.hpp"
#include "drift_gate.hpp"
#include "metrics_timer.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<fs::path> collect_images(const std::string& folder, size_t max_count) {
    std::vector<fs::path> paths;
    if (!fs::exists(folder)) {
        return paths;
    }
    for (const auto& entry : fs::recursive_directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());
    if (max_count > 0 && paths.size() > max_count) {
        paths.resize(max_count);
    }
    return paths;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path = "artifacts/openvino_mixed_int8/patchcore_mvtec_ad_screw.xml";
    std::string images_dir = "data/raw/mvtec_ad/screw/test";
    std::string drift_json = "artifacts/calibration/drift_baseline_mvtec_ad_screw.json";
    std::string conformal_json = "artifacts/calibration/conformal_patchcore_mvtec_ad_screw.json";
    std::string mode = "latency";
    size_t limit = 60;

    for (int i = 1; i + 1 < argc; i += 2) {
        std::string flag = argv[i];
        std::string val = argv[i + 1];
        if (flag == "--model") model_path = val;
        else if (flag == "--images") images_dir = val;
        else if (flag == "--drift") drift_json = val;
        else if (flag == "--conformal") conformal_json = val;
        else if (flag == "--mode") mode = val;
        else if (flag == "--limit") limit = static_cast<size_t>(std::stoul(val));
    }

    std::cout << "=== EdgeLogic-AOI C++17 OpenVINO Edge Engine ===\n";
    std::cout << "Model : " << model_path << "\n";
    std::cout << "Images: " << images_dir << "\n";
    std::cout << "Mode  : " << mode << "\n\n";

    edgelogic::DriftGate drift_gate;
    if (drift_gate.load_baseline(drift_json)) {
        std::cout << "[OK] Drift Gate baseline loaded: " << drift_json << "\n";
    } else {
        std::cout << "[WARN] Drift Gate baseline not found (running passthrough).\n";
    }

    edgelogic::ConformalTriage triage;
    if (triage.load_from_json(conformal_json)) {
        std::cout << "[OK] Conformal thresholds loaded: t_pass="
                  << triage.thresholds().t_pass << ", t_reject="
                  << triage.thresholds().t_reject << "\n";
    } else {
        std::cout << "[WARN] Using default Conformal thresholds.\n";
    }

    edgelogic::AOIEngine engine;
    edgelogic::EngineConfig cfg;
    cfg.model_xml_path = model_path;
    cfg.device = "CPU";
    cfg.performance_hint = (mode == "throughput") ? "THROUGHPUT" : "LATENCY";
    if (!engine.load(cfg)) {
        return 1;
    }
    std::cout << "[OK] OpenVINO compiled on CPU (optimal async requests: "
              << engine.optimal_num_requests() << ")\n\n";

    auto paths = collect_images(images_dir, limit);
    if (paths.empty()) {
        std::cerr << "[ERROR] No images found in: " << images_dir << "\n";
        return 1;
    }

    std::vector<cv::Mat> frames;
    frames.reserve(paths.size());
    for (const auto& p : paths) {
        cv::Mat img = cv::imread(p.string(), cv::IMREAD_COLOR);
        if (!img.empty()) frames.push_back(img);
    }

    if (mode == "throughput") {
        auto stats = engine.benchmark_async(frames);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "--- Async Multi-Stream Throughput Report ---\n";
        std::cout << "Frames : " << stats.count << "\n";
        std::cout << "P50    : " << stats.p50_ms << " ms\n";
        std::cout << "P95    : " << stats.p95_ms << " ms\n";
        std::cout << "P99    : " << stats.p99_ms << " ms\n";
        std::cout << "FPS    : " << stats.fps << " FPS (Async)\n";
        return 0;
    }

    // Warmup 3 frames
    for (size_t i = 0; i < std::min<size_t>(3, frames.size()); ++i) {
        engine.infer_single(frames[i]);
    }

    edgelogic::MetricsTimer timer;
    size_t cnt_pass = 0, cnt_rev = 0, cnt_rej = 0, cnt_halt = 0;

    for (size_t i = 0; i < frames.size(); ++i) {
        auto drift = drift_gate.check(frames[i]);
        if (!drift.passed) {
            ++cnt_halt;
            continue;
        }
        auto res = engine.infer_single(frames[i]);
        timer.record_ms(res.infer_ms);
        auto decision = triage.decide(res.anomaly_score);

        if (decision == edgelogic::TriageDecision::PASS) ++cnt_pass;
        else if (decision == edgelogic::TriageDecision::UNCERTAIN_REVIEW) ++cnt_rev;
        else ++cnt_rej;

        if (i < 10) {
            std::cout << std::fixed << std::setprecision(4)
                      << "[" << std::setw(2) << i + 1 << "] "
                      << paths[i].parent_path().filename().string() << "/"
                      << paths[i].filename().string()
                      << " | score=" << res.anomaly_score
                      << " | " << edgelogic::ConformalTriage::to_string(decision)
                      << " (" << edgelogic::ConformalTriage::to_plc_hex(decision) << ")"
                      << " | " << std::setprecision(2) << res.infer_ms << " ms\n";
        }
    }

    auto stats = timer.summarize();
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\n--- EdgeLogic-AOI Latency & PLC Summary ---\n";
    std::cout << "Processed : " << frames.size() << " parts\n";
    std::cout << "PLC Counts: Pass(0x01)=" << cnt_pass
              << " | Review(0x02)=" << cnt_rev
              << " | Reject(0x03)=" << cnt_rej
              << " | DriftHalt(0xFF)=" << cnt_halt << "\n";
    std::cout << "Latency   : Mean=" << stats.mean_ms
              << " ms | P50=" << stats.p50_ms
              << " ms | P95=" << stats.p95_ms
              << " ms | P99=" << stats.p99_ms << " ms\n";
    std::cout << "Throughput: " << stats.fps << " FPS (Single-Stream)\n";
    return 0;
}