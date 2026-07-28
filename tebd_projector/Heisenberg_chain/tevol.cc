#include <nlohmann/json.hpp>
#include <itensor/all_mps.h>
#include <random>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <string>
#include "../tnmc_tebd.hpp"

int main() {
    nlohmann::json setting;
    std::ifstream input("setting.json");
    input >> setting;

    int N = setting["LatticeSize"];
    int chi = setting["BondDimension"];
    int Nt = setting["StepNumber"];
    double Jz = setting["Jz"];
    double dt = setting["TimeStep"];
    double pinv_threshold = setting["PseudoInvThreshold"];
    double random_ratio = setting["RandomRatio"];
    double gamma = setting["DecayConstant"];
    double power = setting.value("SingularValuePower", 1.0);
    int n_warm = setting["WarmingStep"];
    int n_sample = setting["SampleNumber"];
    int interval = setting["Interval"];

    itensor::SpinHalf sites(N, {"ConserveQNs", false});
    auto state = itensor::InitState(sites);
    for (int i = 1; i <= N; ++i) {
        if (i%2 == 1) {
            state.set(i, "Up");
        } else {
            state.set(i, "Dn");
        }
    }
    auto psi = itensor::MPS(state);

    std::vector<std::vector<std::pair<int, itensor::ITensor>>> gates;
    gates.reserve(3*(2*Nt+1));
    std::vector<std::pair<int, itensor::ITensor>> even_xx_layer, even_yy_layer, even_zz_layer, odd_xx_layer, odd_yy_layer, odd_zz_layer, odd_xx_half_layer, odd_yy_half_layer, odd_zz_half_layer;
    even_xx_layer.reserve(N/2);
    even_yy_layer.reserve(N/2);
    even_zz_layer.reserve(N/2);
    odd_xx_layer.reserve(N/2);
    odd_yy_layer.reserve(N/2);
    odd_zz_layer.reserve(N/2);
    odd_xx_half_layer.reserve(N/2);
    odd_yy_half_layer.reserve(N/2);
    odd_zz_half_layer.reserve(N/2);
    std::complex<double> ii(0.0, 1.0);
    for (int i = 1; i < N; i+=2) {
        itensor::ITensor xx_gate = std::cos(0.25*dt)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.25*dt)*itensor::op(sites, "Sx", i)*itensor::op(sites, "Sx", i+1);
        itensor::ITensor xx_gate_half = std::cos(0.125*dt)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.125*dt)*itensor::op(sites, "Sx", i)*itensor::op(sites, "Sx", i+1);
        itensor::ITensor yy_gate = std::cos(0.25*dt)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.25*dt)*itensor::op(sites, "Sy", i)*itensor::op(sites, "Sy", i+1);
        itensor::ITensor yy_gate_half = std::cos(0.125*dt)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.125*dt)*itensor::op(sites, "Sy", i)*itensor::op(sites, "Sy", i+1);
        itensor::ITensor zz_gate = std::cos(0.25*dt*Jz)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.25*dt*Jz)*itensor::op(sites, "Sz", i)*itensor::op(sites, "Sz", i+1);
        itensor::ITensor zz_gate_half = std::cos(0.125*dt*Jz)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.125*dt*Jz)*itensor::op(sites, "Sz", i)*itensor::op(sites, "Sz", i+1);
        odd_xx_layer.emplace_back(i, xx_gate);
        odd_yy_layer.emplace_back(i, yy_gate);
        odd_zz_layer.emplace_back(i, zz_gate);
        odd_xx_half_layer.emplace_back(i, xx_gate_half);
        odd_yy_half_layer.emplace_back(i, yy_gate_half);
        odd_zz_half_layer.emplace_back(i, zz_gate_half);
    }

    for (int i = 2; i < N; i+=2) {
        itensor::ITensor xx_gate = std::cos(0.25*dt)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.25*dt)*itensor::op(sites, "Sx", i)*itensor::op(sites, "Sx", i+1);
        itensor::ITensor yy_gate = std::cos(0.25*dt)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.25*dt)*itensor::op(sites, "Sy", i)*itensor::op(sites, "Sy", i+1);
        itensor::ITensor zz_gate = std::cos(0.25*dt*Jz)*itensor::op(sites, "Id", i)*itensor::op(sites, "Id", i+1)
            - 4.0*ii*std::sin(0.25*dt*Jz)*itensor::op(sites, "Sz", i)*itensor::op(sites, "Sz", i+1);
        even_xx_layer.emplace_back(i, xx_gate);
        even_yy_layer.emplace_back(i, yy_gate);
        even_zz_layer.emplace_back(i, zz_gate);
    }

    gates.push_back(odd_xx_half_layer);
    gates.push_back(odd_yy_half_layer);
    gates.push_back(odd_zz_half_layer);
    gates.push_back(even_xx_layer);
    gates.push_back(even_yy_layer);
    gates.push_back(even_zz_layer);
    for (int t = 1; t < Nt; ++t) {
        gates.push_back(odd_xx_layer);
        gates.push_back(odd_yy_layer);
        gates.push_back(odd_zz_layer);

        gates.push_back(even_xx_layer);
        gates.push_back(even_yy_layer);
        gates.push_back(even_zz_layer);
    }
    gates.push_back(odd_xx_half_layer);
    gates.push_back(odd_yy_half_layer);
    gates.push_back(odd_zz_half_layer);

    tnmc_tebd::tnmc_tebd tevol(gates, psi, chi, pinv_threshold, random_ratio, power);

    std::vector<itensor::MPO> ops(7);
    itensor::AutoMPO ampo(sites);
    for (int i = 1; i < N; ++i) {
        if (i < N) {
            ampo += Jz, "Sz", i, "Sz", i+1;
            ampo += 0.5,"S+", i, "S-", i+1;
            ampo += 0.5,"S-", i, "S+", i+1;
        }
    }
    ops.at(0) = itensor::toMPO(ampo);

    for (int i = 1; i <= 3; ++i) {
        itensor::AutoMPO ampo(sites);
        ampo += "Sz", N/2, "Sz", N/2+i;
        ops.at(i) = itensor::toMPO(ampo);
    }
    for (int i = 1; i <= 3; ++i) {
        itensor::AutoMPO ampo(sites);
        ampo += "Sx", N/2, "Sx", N/2+i;
        ops.at(3+i) = itensor::toMPO(ampo);
    }

    nlohmann::json output;
    output["setting"] = setting;

    std::cout << "TEBD log overlap: " << tevol.log_overlap_ket() << std::endl;
    std::function<double(double, double)> log_wf;
    if (gamma <= 0.0) {
        log_wf = [](double log_bra, double log_ket) -> double { return 0.5*(log_bra + log_ket); };
    } else {
        double log_gamma = std::log(gamma) - 2.0*tevol.log_overlap_ket();
        log_wf = [log_gamma](double log_bra, double log_ket) -> double { return 0.5*(log_bra + log_ket) - std::exp(log_gamma + log_ket + log_bra); };
    }
    tevol.set_log_weight_function(log_wf);

    std::random_device rd;
    uint64_t seed = rd();
    seed = (seed << 32) | rd();
    std::stringstream stream;
    stream << std::uppercase << std::setfill('0') << std::setw(16) << std::hex << seed;
    std::string seed_str = stream.str();
    output["seed"] = seed_str;
    std::string output_name = "sample_" + seed_str + ".mpac";
    std::mt19937_64 engine(seed);
    std::vector<double> accept, skip, random, log_weights, overlap_abs, overlap_arg, denom;
    std::vector<double> ene,
                        zz_1, zz_2, zz_3,
                        xx_1, xx_2, xx_3;
    accept.reserve(n_sample);
    skip.reserve(n_sample);
    random.reserve(n_sample);
    log_weights.reserve(n_sample);
    denom.reserve(n_sample);
    overlap_abs.reserve(n_sample);
    overlap_arg.reserve(n_sample);
    ene.reserve(n_sample);
    zz_1.reserve(n_sample);
    zz_2.reserve(n_sample);
    zz_3.reserve(n_sample);
    xx_1.reserve(n_sample);
    xx_2.reserve(n_sample);
    xx_3.reserve(n_sample);
    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < n_warm; ++i) {
        tevol.MH_update(engine);
    }
    auto end = std::chrono::system_clock::now();
    double avg_elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::seconds>(end - start).count()) / n_warm;
    output["required_time"] = avg_elapsed;
    std::cout << "1 sample requires " << avg_elapsed << "s" << std::endl;
    start = std::chrono::system_clock::now();
    for (int i = 0; i < n_sample; ++i) {
        auto ratio = tevol.MH_update(engine);
        auto expectations = tevol.expectation_values(ops);
        accept.push_back(ratio.at(0));
        skip.push_back(ratio.at(1));
        random.push_back(ratio.at(2));
        ene.push_back(expectations.at(1).real());
        zz_1.push_back(expectations.at(2).real());
        zz_2.push_back(expectations.at(3).real());
        zz_3.push_back(expectations.at(4).real());
        xx_1.push_back(expectations.at(5).real());
        xx_2.push_back(expectations.at(6).real());
        xx_3.push_back(expectations.at(7).real());
        log_weights.push_back(tevol.log_weight());
        denom.push_back(expectations.at(0).real());
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
            output["random_ratio"] = random;
            output["skip_ratio"] = skip;
            output["sampled_log_weight"] = log_weights;
            output["sampled_denom"] = denom;
            output["sampled_overlap_abs"] = overlap_abs;
            output["sampled_overlap_arg"] = overlap_arg;
            output["sampled_energy"] = ene;
            output["sampled_zz1"] = zz_1;
            output["sampled_zz2"] = zz_2;
            output["sampled_zz3"] = zz_3;
            output["sampled_xx1"] = xx_1;
            output["sampled_xx2"] = xx_2;
            output["sampled_xx3"] = xx_3;

            std::ofstream of(output_name, std::ios::out | std::ios::binary);
            std::vector<std::uint8_t> binary = nlohmann::json::to_msgpack(output);
            of.write(reinterpret_cast<const char *>(binary.data()), binary.size());
            of.close();
            start = std::chrono::system_clock::now();
        }
    }
    output["acceptance_ratio"] = accept;
    output["skip_ratio"] = skip;
    output["random_ratio"] = random;
    output["sampled_log_weight"] = log_weights;
    output["sampled_denom"] = denom;
    output["sampled_overlap_abs"] = overlap_abs;
    output["sampled_overlap_arg"] = overlap_arg;
    output["sampled_energy"] = ene;
    output["sampled_zz1"] = zz_1;
    output["sampled_zz2"] = zz_2;
    output["sampled_zz3"] = zz_3;
    output["sampled_xx1"] = xx_1;
    output["sampled_xx2"] = xx_2;
    output["sampled_xx3"] = xx_3;
    std::ofstream of(output_name, std::ios::out | std::ios::binary);
    std::vector<std::uint8_t> binary = nlohmann::json::to_msgpack(output);
    of.write(reinterpret_cast<const char *>(binary.data()), binary.size());
    of.close();
    
    return 0;
}
