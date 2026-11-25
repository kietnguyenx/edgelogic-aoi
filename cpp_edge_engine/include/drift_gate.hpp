#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace edgelogic {

struct DriftBaseline {
    int image_size = 256;
    int hist_bins = 32;
    double blur_min_threshold = 30.0;
    double brightness_min = 40.0;
    double brightness_max = 220.0;
    double contrast_min_threshold = 15.0;
    double bhattacharyya_max = 0.70;
    double clipped_pixel_ratio_max = 0.15;
    std::vector<float> reference_hist;
};

struct DriftResult {
    bool passed = true;
    double laplacian_var = 0.0;
    double brightness_mean = 0.0;
    double contrast_std = 0.0;
    double clipped_ratio = 0.0;
    double bhattacharyya_dist = 0.0;
    std::vector<std::string> flags;
};

class DriftGate {
public:
    DriftGate() = default;

    bool load_baseline(const std::string& json_path);
    bool is_loaded() const { return loaded_; }
    DriftResult check(const cv::Mat& frame_bgr) const;

private:
    DriftBaseline baseline_;
    bool loaded_ = false;
};

}  // namespace edgelogic