#include "conformal_triage.hpp"

#include <fstream>
#include <regex>
#include <sstream>

namespace edgelogic {

namespace {

bool extract_json_number(const std::string& content, const std::string& key, double& out_val) {
    std::regex pattern("\"" + key + "\"\\s*:\\s*([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)");
    std::smatch match;
    if (std::regex_search(content, match, pattern) && match.size() >= 2) {
        out_val = std::stod(match[1].str());
        return true;
    }
    return false;
}

}  // namespace

ConformalTriage::ConformalTriage(const ConformalThresholds& thresholds)
    : thresholds_(thresholds) {}

bool ConformalTriage::load_from_json(const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string content = buffer.str();

    double tp = thresholds_.t_pass;
    double tr = thresholds_.t_reject;
    bool ok_pass = extract_json_number(content, "t_pass", tp);
    bool ok_rej = extract_json_number(content, "t_reject", tr);
    if (ok_pass && ok_rej) {
        thresholds_.t_pass = tp;
        thresholds_.t_reject = tr;
        return true;
    }
    return false;
}

TriageDecision ConformalTriage::decide(double anomaly_score) const {
    if (anomaly_score < thresholds_.t_pass) {
        return TriageDecision::PASS;
    }
    if (anomaly_score > thresholds_.t_reject) {
        return TriageDecision::DEFECT_REJECT;
    }
    return TriageDecision::UNCERTAIN_REVIEW;
}

std::string ConformalTriage::to_string(TriageDecision decision) {
    switch (decision) {
        case TriageDecision::PASS:             return "PASS";
        case TriageDecision::UNCERTAIN_REVIEW: return "UNCERTAIN_REVIEW";
        case TriageDecision::DEFECT_REJECT:    return "DEFECT_REJECT";
        case TriageDecision::DRIFT_HALT:       return "DRIFT_HALT";
        default:                               return "UNKNOWN";
    }
}

std::string ConformalTriage::to_plc_hex(TriageDecision decision) {
    switch (decision) {
        case TriageDecision::PASS:             return "0x01";
        case TriageDecision::UNCERTAIN_REVIEW: return "0x02";
        case TriageDecision::DEFECT_REJECT:    return "0x03";
        case TriageDecision::DRIFT_HALT:       return "0xFF";
        default:                               return "0x00";
    }
}

}  // namespace edgelogic