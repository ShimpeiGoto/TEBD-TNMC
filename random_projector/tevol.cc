#include <nlohmann/json.hpp>
#include <itensor/all_mps.h>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include "tnmc_random_unitary.hpp"

int main() {
    nlohmann::json setting;
    std::ifstream input("setting.json");
    input >> setting;

    int N = setting["LatticeSize"];
    int chi = setting["BondDimension"];
    int Nt = setting["StepNumber"];
    double hx = setting["TransverseField"];
    double hz = setting["LongitudinalField"];
    double gamma = setting.value("DecayConstant", 0.0);
    int n_warm = setting["WarmingStep"];
    int n_sample = setting["SampleNumber"];
    int interval = setting["Interval"];
    hx *= std::acos(-1.0);
    hz *= std::acos(-1.0);

    itensor::SpinHalf sites(N, {"ConserveQNs", false});
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= N; ++i) {
        state.set(i, "Up");
    }
    auto psi = itensor::MPS(state);
    double inv_sqrt2 = 0.5*std::sqrt(2.0);
    std::complex<double> ii(0.0, 1.0);
    itensor::ITensor initial_gate = inv_sqrt2*(itensor::op(sites, "Id", 1) - 2.0*ii*itensor::op(sites, "Sy", 1));
    psi.Aref(1) *= initial_gate;
    psi.Aref(1).noPrime("Site");
    for (int n = 2; n < N; n+=2) {
        auto AA = psi.A(n) * psi.A(n+1);
        AA *= 2.0*inv_sqrt2*(itensor::op(sites, "Sx", n) + itensor::op(sites, "Sz", n));
        AA.noPrime("Site");
        itensor::ITensor cnot = itensor::op(sites, "projUp", n)*itensor::op(sites, "Id", n+1) + 2.0*itensor::op(sites, "projDn", n)*itensor::op(sites, "Sx", n+1);
        AA *= cnot;
        AA.noPrime("Site");
        auto [left, right] = itensor::factor(AA, {sites(n), itensor::leftLinkIndex(psi, n)}, {"Tags", tinyformat::format("Link,l=%d", n)});
        psi.Aref(n) = left;
        psi.Aref(n+1) = right;
    }

    std::vector<std::pair<int, itensor::ITensor>> zz_layer;
    zz_layer.reserve(N);
    for (int i = 1; i < N; ++i) {
        itensor::ITensor first_zz_gate = inv_sqrt2*(itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1) - 4.0*ii*itensor::op(sites, "Sz", i)*itensor::op(sites, "Sz", i+1));
        itensor::ITensor second_zz_gate = first_zz_gate;
        itensor::ITensor z_gate = std::cos(hz)*itensor::op(sites, "Id", i+1) - 2.0*ii*std::sin(hz)*itensor::op(sites, "Sz", i+1);
        auto x_gate_left = std::cos(hx)*itensor::op(sites, "Id", i) - 2.0*ii*std::sin(hx)*itensor::op(sites, "Sx", i);
        auto x_gate_right = std::cos(hx)*itensor::op(sites, "Id", i+1) - 2.0*ii*std::sin(hx)*itensor::op(sites, "Sx", i+1);
        first_zz_gate.prime("Site");
        first_zz_gate *= z_gate;
        first_zz_gate.mapPrime(1, 0);
        first_zz_gate.mapPrime(2, 1);
        first_zz_gate *= itensor::prime(x_gate_left);
        first_zz_gate.mapPrime(2, 1);

        second_zz_gate.prime("Site");
        second_zz_gate *= x_gate_right;
        second_zz_gate.mapPrime(1, 0);
        second_zz_gate.mapPrime(2, 1);
        second_zz_gate *= itensor::prime(z_gate);
        second_zz_gate.mapPrime(2, 1);

        first_zz_gate *= itensor::prime(second_zz_gate);
        first_zz_gate.mapPrime(2, 1);
        zz_layer.emplace_back(i, first_zz_gate);
    }

    std::vector<std::vector<std::pair<int, itensor::ITensor>>> gates;
    gates.reserve(Nt);
    std::vector<int> support {Nt+1};
    for (int t = Nt; t > 0; --t) {
        int start = t%2 == 1 ? 1 : 2;
        std::vector<std::pair<int, itensor::ITensor>> layer;
        for (int n = start; n < N; n+=2) {
            int left_idx = zz_layer.at(n-1).first;
            bool is_left_contained = (std::find(support.begin(), support.end(), left_idx) != support.end());
            bool is_right_contained = (std::find(support.begin(), support.end(), left_idx+1) != support.end());
            if (is_left_contained or is_right_contained) {
                layer.push_back(zz_layer.at(n-1));
                if (not is_left_contained) {
                    support.push_back(left_idx);
                }
                if (not is_right_contained) {
                    support.push_back(left_idx+1);
                }
            }
        }
        gates.push_back(layer);
    }
    std::reverse(gates.begin(), gates.end());

    nlohmann::json output;
    output["setting"] = setting;
    std::random_device rd;
    uint64_t seed = rd();
    seed = (seed << 32) | rd();
    std::stringstream stream;
    stream << std::uppercase << std::setfill('0') << std::setw(16) << std::hex << seed;
    std::string seed_str = stream.str();
    output["seed"] = seed_str;
    std::string output_name = "sample_" + seed_str + ".mpac";
    std::mt19937_64 engine(seed);
    std::function<double(double, double)> log_wf = [gamma](double bra, double ket) -> double { return 0.5*(std::log(bra) + std::log(ket)) - gamma*bra*ket; };
    tnmc_random::tnmc_random tevol(gates, psi, chi, engine);
    std::vector<itensor::MPO> ops(1);
    itensor::AutoMPO ampo(sites);
    ampo += 2.0, "Sx", Nt+1;
    ops.at(0) = itensor::toMPO(ampo);
    std::cout << std::setprecision(16);
    std::vector<double> trg_expectations;
    trg_expectations.reserve(1);
    std::vector<itensor::MPO> mpo_vec(1);
    auto result = tevol.expectation_values(ops);
    double trg_denom = result.at(0).real();
    std::cout << trg_denom << std::endl;
    std::cout << result.at(1).real() / trg_denom << std::endl;
    output["trg_overlap"] = trg_denom;
    output["trg_expectations"] = result.at(1).real() / trg_denom;
    output["trg_log_weight"] = tevol.log_weight();

    tevol.set_log_weight_function(log_wf);

    std::vector<double> accept, skip, log_weights;
    std::vector<double> X_obs, denom;
    accept.reserve(n_sample);
    skip.reserve(n_sample);
    X_obs.reserve(n_sample);
    log_weights.reserve(n_sample);
    denom.reserve(n_sample);
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < n_warm; ++i) {
        tevol.MH_update();
    }
    auto end = std::chrono::system_clock::now();
    double avg_elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(end - start).count()) / n_warm;
    output["required_time"] = avg_elapsed;
    std::cout << "1 sample requires " << avg_elapsed << "s" << std::endl;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < n_sample; ++i) {
        auto ratio = tevol.MH_update();
        auto result = tevol.expectation_values(ops);
        accept.push_back(ratio.at(0));
        skip.push_back(ratio.at(1));
        X_obs.push_back(result.at(1).real());
        log_weights.push_back(tevol.log_weight());
        denom.push_back(result.at(0).real());
        if (i > 0 and i % interval == 0) {
            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
            std::cout << "Elapsed time for 1 sample: " << static_cast<double>(duration) / interval << "s" << std::endl;
            int est_remain = static_cast<int>(duration) * (n_sample - i) / interval;
            std::cout << "Estimated remaining time: ";
            if (est_remain >= 3600) {
                int hour = est_remain / 3600;
                std::cout << hour << "h ";
                est_remain -= hour*3600;
            }
            if (est_remain >= 60) {
                int minutes = est_remain / 60;
                std::cout << minutes << "m ";
                est_remain -= minutes*60;
            }
            std::cout << est_remain << "s" << std::endl;
            output["acceptance_ratio"] = accept;
            output["skip_ratio"] = skip;
            output["sampled_Xexp"] = X_obs;
            output["sampled_log_weight"] = log_weights;
            output["sampled_denom"] = denom;

            std::ofstream of(output_name, std::ios::out | std::ios::binary);
            std::vector<std::uint8_t> binary = nlohmann::json::to_msgpack(output);
            of.write(reinterpret_cast<const char *>(binary.data()), binary.size());
            of.close();
            start = std::chrono::system_clock::now();
        }
    }
    output["acceptance_ratio"] = accept;
    output["skip_ratio"] = skip;
    output["sampled_Xexp"] = X_obs;
    output["sampled_log_weight"] = log_weights;
    output["sampled_denom"] = denom;
    std::ofstream of(output_name, std::ios::out | std::ios::binary);
    std::vector<std::uint8_t> binary = nlohmann::json::to_msgpack(output);
    of.write(reinterpret_cast<const char *>(binary.data()), binary.size());
    of.close();
    
    return 0;
}
