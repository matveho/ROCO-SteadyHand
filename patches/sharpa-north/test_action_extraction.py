#!/usr/bin/env python3
"""Test the actual North action-extraction and dispatch method bodies offline.

Uses the SDK's real protobuf classes. Controller, clock, stability checker and
message transport are test doubles; no controller is instantiated or connected.
Production method bodies are extracted verbatim, not rewritten in the test.
"""
import argparse
from pathlib import Path
import subprocess
import tempfile


def method(source, name):
    import re
    match = re.search(r"(?:void|bool) NorthController::" + name + r"\(", source)
    if match is None:
        raise RuntimeError("Missing method: " + name)
    start = match.start()
    # These two methods contain no unmatched braces in strings or comments.
    opening = source.index("{", match.end())
    depth = 1
    for end in range(opening + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start:end + 1]
    raise RuntimeError("Unterminated method: " + name)


PRELUDE = r'''
#include "north.pb.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#define LOG_ERROR(message) do {} while (false)
struct OfflineBus {
    std::vector<std::pair<std::string, std::string>> publications;
    void publish(const std::string& topic, const std::string& payload) {
        publications.emplace_back(topic, payload);
    }
} offline_bus;
#define MESSAGE_SYSTEM offline_bus
double getCurrentTime() { return 1.0; }
template <class Header> void setProtoTimestamp(Header*, double) {}
struct OfflineChecker {
    int calls = 0;
    void action_with_stable_com(bool, float*, float*, float*, float*) { ++calls; }
};
class NorthController {
public:
    std::shared_ptr<OfflineChecker> checker_ = std::make_shared<OfflineChecker>();
    int updates = 0;
    static EXTRACT_RETURN extractActions(const proto::UhrActionBundle&, float[5],
                                         float[7], float[7], float[2]);
    void processActionBundle(proto::UhrActionBundle&);
    void updateActionsFromChecker(proto::UhrActionBundle&) { ++updates; }
};
'''

TESTS = r'''
static int cases = 0;
void require(bool condition, const char* description) {
    if (!condition) throw std::runtime_error(description);
}
struct Outputs {
    float leg[5], left[7], right[7], head[2];
    Outputs() {
        std::fill_n(leg, 5, -991.0f); std::fill_n(left, 7, -992.0f);
        std::fill_n(right, 7, -993.0f); std::fill_n(head, 2, -994.0f);
    }
    void unchanged() const {
        for (float x : leg) require(x == -991.0f, "leg modified on rejection");
        for (float x : left) require(x == -992.0f, "left modified on rejection");
        for (float x : right) require(x == -993.0f, "right modified on rejection");
        for (float x : head) require(x == -994.0f, "head modified on rejection");
    }
};
proto::UhrActionBundle valid_bundle() {
    proto::UhrActionBundle b;
    for (int i = 0; i < 7; ++i) {
        b.mutable_left_arm()->mutable_joint()->add_position(float(i + 1));
        b.mutable_right_arm()->mutable_joint()->add_position(float(-i - 11));
        auto* m = b.mutable_motor()->add_commands();
        m->set_motor_id(i + 1); m->set_command("position"); m->set_value(i + 21);
    }
    for (int i = 0; i < 22; ++i) {
        b.mutable_left_glove()->mutable_joint()->add_position(0.0f);
        b.mutable_right_glove()->mutable_joint()->add_position(0.0f);
    }
    b.mutable_chassis(); b.set_language("offline test");
    return b;
}
bool extract(const proto::UhrActionBundle& b, Outputs& o) {
#ifdef ORIGINAL_BUG
    NorthController::extractActions(b, o.leg, o.left, o.right, o.head);
    return true;
#else
    return NorthController::extractActions(b, o.leg, o.left, o.right, o.head);
#endif
}
void rejected(proto::UhrActionBundle b) {
    Outputs o;
    require(!extract(b, o), "malformed bundle accepted");
    o.unchanged();
    NorthController controller;
    offline_bus.publications.clear();
    controller.processActionBundle(b);
    require(controller.checker_->calls == 0, "invalid bundle reached checker");
    require(controller.updates == 0, "invalid bundle reached correction");
    require(offline_bus.publications.empty(), "invalid bundle was published");
    ++cases;
}
int main() {
    try {
#ifdef ORIGINAL_BUG
        // Separate stack allocations let ASan detect the original first store.
        auto b = valid_bundle();
        float leg[5] = {}, left[7] = {}, right[7] = {}, head[2] = {};
        NorthController::extractActions(b, leg, left, right, head);
        std::cerr << "ERROR: original bug was not detected\n";
        return 2;
#else
        auto b = valid_bundle();
        Outputs o;
        require(extract(b, o), "valid bundle rejected");
        for (int i = 0; i < 7; ++i) {
            require(o.left[6-i] == b.left_arm().joint().position(i), "left order wrong");
            require(o.right[6-i] == b.right_arm().joint().position(i), "right order wrong");
        }
        for (int i = 0; i < 5; ++i) require(o.leg[4-i] == i+21, "leg order changed");
        require(o.head[1] == 26 && o.head[0] == 27, "head order changed");
        ++cases;
        NorthController controller;
        offline_bus.publications.clear();
        controller.processActionBundle(b);
        require(controller.checker_->calls == 1 && controller.updates == 1,
                "valid bundle did not pass through checker");
        require(offline_bus.publications.size() == 7, "valid dispatch count changed");
        for (const auto& p : offline_bus.publications) {
            if (p.first == "arm/action/left" || p.first == "arm/action/right") {
                proto::Arm sent;
                require(sent.ParseFromString(p.second), "invalid serialized arm");
                const auto& expected = p.first == "arm/action/left" ? b.left_arm() : b.right_arm();
                require(sent.joint().SerializeAsString() == expected.joint().SerializeAsString(),
                        "published arm joint order changed");
            }
        }
        ++cases;
        for (int side = 0; side < 2; ++side) {
            for (int size : {0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 64, 1024}) {
                auto bad = valid_bundle();
                auto* arm = side == 0 ? bad.mutable_left_arm() : bad.mutable_right_arm();
                arm->clear_joint();
                for (int i = 0; i < size; ++i) arm->mutable_joint()->add_position(0.0f);
                rejected(bad);
            }
            for (int joint = 0; joint < 7; ++joint) {
                for (float value : {std::numeric_limits<float>::quiet_NaN(),
                                    std::numeric_limits<float>::infinity(),
                                    -std::numeric_limits<float>::infinity()}) {
                    auto bad = valid_bundle();
                    auto* arm = side == 0 ? bad.mutable_left_arm() : bad.mutable_right_arm();
                    arm->mutable_joint()->set_position(joint, value);
                    rejected(bad);
                }
            }
        }
        // Missing arm fields retain existing partial-bundle behavior; a present
        // empty arm was rejected above. No absent arm is fabricated for dispatch.
        for (int missing = 1; missing <= 3; ++missing) {
            auto partial = valid_bundle();
            if (missing & 1) partial.clear_left_arm();
            if (missing & 2) partial.clear_right_arm();
            Outputs unchanged;
            require(extract(partial, unchanged), "absent arm rejected");
            if (missing & 1) for (float x : unchanged.left) require(x == -992.0f, "absent left modified");
            if (missing & 2) for (float x : unchanged.right) require(x == -993.0f, "absent right modified");
            offline_bus.publications.clear();
            controller.processActionBundle(partial);
            for (const auto& p : offline_bus.publications) {
                require(!(missing & 1) || p.first != "arm/action/left", "absent left published");
                require(!(missing & 2) || p.first != "arm/action/right", "absent right published");
            }
            ++cases;
        }
        std::mt19937 generator(173);
        std::uniform_real_distribution<float> values(-3.0f, 3.0f);
        for (int trial = 0; trial < 200; ++trial) {
            auto sample = valid_bundle();
            for (int i = 0; i < 7; ++i) {
                sample.mutable_left_arm()->mutable_joint()->set_position(i, values(generator));
                sample.mutable_right_arm()->mutable_joint()->set_position(i, values(generator));
            }
            Outputs mapped;
            require(extract(sample, mapped), "finite sample rejected");
            for (int i = 0; i < 7; ++i) {
                require(mapped.left[6-i] == sample.left_arm().joint().position(i), "sample left order");
                require(mapped.right[6-i] == sample.right_arm().joint().position(i), "sample right order");
            }
            ++cases;
        }
        std::cout << "PASS: " << cases << " offline cases; no hardware transport\n";
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdk", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expect-original-bug", action="store_true")
    args = parser.parse_args()
    source = args.source.read_text()
    extraction = method(source, "extractActions")
    processing = method(source, "processActionBundle")
    returns = extraction.split()[0]
    if (returns == "void") != args.expect_original_bug:
        parser.error("Source return type does not match requested test mode")
    with tempfile.TemporaryDirectory(prefix="north-offline-test-") as tmp:
        tmp = Path(tmp)
        cpp = tmp / "test.cpp"
        cpp.write_text(PRELUDE.replace("EXTRACT_RETURN", returns) + "\n" +
                       extraction + "\n" + processing + "\n" + TESTS)
        command = ["nice", "-n", "15", "c++", "-std=c++17", "-O1", "-g",
                   "-fsanitize=address,undefined", "-fno-omit-frame-pointer",
                   "-fno-sanitize-recover=all", "-I" + str(args.sdk / "build"),
                   str(cpp), str(args.sdk / "build/north.pb.cc"), "-lprotobuf",
                   "-pthread", "-o", str(tmp / "test")]
        if args.expect_original_bug:
            command.insert(5, "-DORIGINAL_BUG")
        subprocess.run(command, check=True)
        result = subprocess.run([str(tmp / "test")], capture_output=True, text=True)
        if args.expect_original_bug:
            if result.returncode == 0 or "AddressSanitizer: stack-buffer-overflow" not in result.stderr:
                print(result.stdout + result.stderr)
                raise SystemExit("Expected original stack-buffer-overflow was not reproduced")
            print("PASS: unmodified original reproduces AddressSanitizer stack-buffer-overflow")
            print("\n".join(result.stderr.splitlines()[:8]))
        else:
            print(result.stdout + result.stderr, end="")
            if result.returncode:
                raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
