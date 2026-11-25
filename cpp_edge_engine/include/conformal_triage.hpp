#pragma once

#include <cstdint>
#include <string>

namespace edgelogic {

enum class TriageDecision : uint8_t {
    PASS = 0x01,
    UNCERTAIN_REVIEW = 0x02,
    DEFECT_REJECT = 0x03,
    DRIFT_HALT = 0xFF
};

struct ConformalThresholds {
    double t_pass = 0.40;
    double t_reject = 0.60;
};

class ConformalTriage {
public:
    ConformalTriage() = default;
    explicit ConformalTriage(const ConformalThresholds& thresholds);

    bool load_from_json(const std::string& json_path);
    TriageDecision decide(double anomaly_score) const;
    const ConformalThresholds& thresholds() const { return thresholds_; }

    static std::string to_string(TriageDecision decision);
    static std::string to_plc_hex(TriageDecision decision);

private:
    ConformalThresholds thresholds_;
};

}  // namespace edgelogic
