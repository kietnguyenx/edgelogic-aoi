#include "drift_gate.hpp"

#include <opencv2/imgproc.hpp>

#include <fstream>
#include <regex>
#include <sstream>

namespace edgelogic {

namespace {

bool extract_number(const std::string& content, const std::string& key, double& out_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(content, match, pattern) && match.size() >= 2) {
        out_val = std::stod(match[1].str());
        return true;
    }
    return false;
}

std::vector<float> extract_float_array(const std::string& content, const std::string& key) {
    std::vector<float> values;
    std::regex arr_pat("\"" + key + "\"\\s*:\\s*\\[([^\\]]*)\\]");
    std::smatch match;
    if (!std::regex_search(content, match, arr_pat) || match.size() < 2) {
        return values;
    }
    std::string body = match[1].str();
    std::regex num_pat("[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?");
    auto it = std::sregex_iterator(body.begin(), body.end(), num_pat);
    auto end = std::sregex_iterator();
    for (; it != end; ++it) {
        values.push_back(std::stof(it->str()));
    }
    return values;
}

}  // namespace

bool DriftGate::load_baseline(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        loaded_ = false;
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string content = buffer.str();

    double tmp = 0.0;
    if (extract_number(content, "image_size", tmp)) baseline_.image_size = static_cast<int>(tmp);
    if (extract_number(content, "hist_bins", tmp)) baseline_.hist_bins = static_cast<int>(tmp);
    extract_number(content, "blur_min_threshold", baseline_.blur_min_threshold);
    extract_number(content, "brightness_min", baseline_.brightness_min);
    extract_number(content, "brightness_max", baseline_.brightness_max);
    extract_number(content, "contrast_min_threshold", baseline_.contrast_min_threshold);
    extract_number(content, "bhattacharyya_max", baseline_.bhattacharyya_max);
    extract_number(content, "clipped_pixel_ratio_max", baseline_.clipped_pixel_ratio_max);
    baseline_.reference_hist = extract_float_array(content, "reference_hist");

    loaded_ = !baseline_.reference_hist.empty();
    return loaded_;
}
DriftResult DriftGate::check(const cv::Mat& frame_bgr) const {
    DriftResult res;
    if (frame_bgr.empty()) {
        res.passed = false;
        res.flags.push_back("EMPTY_FRAME");
        return res;
    }

    cv::Mat resized, gray;
    cv::resize(frame_bgr, resized, cv::Size(baseline_.image_size, baseline_.image_size), 0, 0, cv::INTER_AREA);
    if (resized.channels() == 3) {
        cv::cvtColor(resized, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = resized;
    }

    // 1. Sharpness (Laplacian variance)
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar lap_mean, lap_std;
    cv::meanStdDev(lap, lap_mean, lap_std);
    res.laplacian_var = lap_std[0] * lap_std[0];

    // 2. Brightness & Contrast
    cv::Scalar mean_val, std_val;
    cv::meanStdDev(gray, mean_val, std_val);
    res.brightness_mean = mean_val[0];
    res.contrast_std = std_val[0];

    // 3. Sensor clipping (<=2 or >=253)
    cv::Mat clipped_mask = (gray <= 2) | (gray >= 253);
    res.clipped_ratio = static_cast<double>(cv::countNonZero(clipped_mask)) /
                        static_cast<double>(gray.rows * gray.cols);

    // 4. Bhattacharyya histogram distance
    if (!baseline_.reference_hist.empty()) {
        int hist_size[] = {baseline_.hist_bins};
        float range[] = {0.0f, 256.0f};
        const float* ranges[] = {range};
        int channels[] = {0};
        cv::Mat hist;
        cv::calcHist(&gray, 1, channels, cv::Mat(), hist, 1, hist_size, ranges, true, false);
        hist /= static_cast<float>(gray.rows * gray.cols);

        cv::Mat ref_hist(static_cast<int>(baseline_.reference_hist.size()), 1, CV_32F,
                         const_cast<float*>(baseline_.reference_hist.data()));
        res.bhattacharyya_dist = cv::compareHist(hist, ref_hist, cv::HISTCMP_BHATTACHARYYA);
    }

    // 5. Evaluate interlock thresholds
    if (loaded_) {
        if (res.laplacian_var < baseline_.blur_min_threshold)
            res.flags.push_back("CAMERA_BLUR_OUT_OF_FOCUS");
        if (res.brightness_mean < baseline_.brightness_min)
            res.flags.push_back("LIGHTING_TOO_DARK");
        if (res.brightness_mean > baseline_.brightness_max)
            res.flags.push_back("LIGHTING_OVEREXPOSED");
        if (res.contrast_std < baseline_.contrast_min_threshold)
            res.flags.push_back("LOW_CONTRAST_WASHED_OUT");
        if (res.clipped_ratio > baseline_.clipped_pixel_ratio_max)
            res.flags.push_back("SENSOR_CLIPPING_GLARE");
        if (res.bhattacharyya_dist > baseline_.bhattacharyya_max)
            res.flags.push_back("ILLUMINATION_HISTOGRAM_DRIFT");
    }

    res.passed = res.flags.empty();
    return res;
}

}  // namespace edgelogic

