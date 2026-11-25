#pragma once

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

namespace edgelogic {

struct LatencyStats {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double fps = 0.0;
    size_t count = 0;
};

class MetricsTimer {
public:
    using Clock = std::chrono::steady_clock;

    void start() {
        t_start_ = Clock::now();
    }

    double stop_ms() {
        auto t_end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_end - t_start_).count();
        samples_ms_.push_back(ms);
        return ms;
    }

    void record_ms(double ms) {
        samples_ms_.push_back(ms);
    }

    void reset() {
        samples_ms_.clear();
    }
    LatencyStats summarize(double total_wall_ms = 0.0) const {
        LatencyStats stats;
        stats.count = samples_ms_.size();
        if (samples_ms_.empty()) {
            return stats;
        }

        std::vector<double> sorted = samples_ms_;
        std::sort(sorted.begin(), sorted.end());

        double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
        stats.mean_ms = sum / static_cast<double>(sorted.size());
        stats.p50_ms = percentile(sorted, 0.50);
        stats.p95_ms = percentile(sorted, 0.95);
        stats.p99_ms = percentile(sorted, 0.99);

        if (total_wall_ms > 0.0) {
            stats.fps = (static_cast<double>(sorted.size()) * 1000.0) / total_wall_ms;
        } else if (stats.mean_ms > 0.0) {
            stats.fps = 1000.0 / stats.mean_ms;
        }
        return stats;
    }

private:
    static double percentile(const std::vector<double>& sorted, double q) {
        if (sorted.empty()) return 0.0;
        double idx = q * static_cast<double>(sorted.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = std::min(lo + 1, sorted.size() - 1);
        double frac = idx - static_cast<double>(lo);
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    }

    Clock::time_point t_start_;
    std::vector<double> samples_ms_;
};

}  // namespace edgelogic