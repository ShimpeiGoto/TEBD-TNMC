/**
@file tnmc_random_unitary.hpp
@brief Header file for TEBD-based tensor network Monte Carlo with random projectors
*/
#pragma once
#include <algorithm>
#include <iterator>
#include <vector>
#include <random>
#include <itensor/all_mps.h>
#include "itensor/itensor.h"
#include "itensor/mps/mpo.h"
#include "tnmc_projector.hpp"
#include "tensor_storage.hpp"

/**
@namespace tnmc_tebd
@brief Namespace containing the class that manages TEBD-based tensor network Monte Carlo
*/
namespace tnmc_random {
    /**
    @brief Enumerator representing the direction of local overlap update
    */
    enum class direction {
        to_left, ///< Direction to left
        to_right ///< Direction to right
    };

    /**
     @brief Function that generates unitary matrix Haar randomly following F. Mezzadri, NOTICES of the AMS 54, 592
     @param [in] std::vector<itensor::Index> inds: Set of indices used in a result tensor
     @param [inout] std::mt19937_64& engine :  random number generator
     @return Haar-randomly generated unitary matrix. The structure of returned tensor is Tensor(u_inds, prime(u_inds))
     */
    inline itensor::ITensor random_unitary(std::vector<itensor::Index> inds, std::mt19937_64 &engine) {
        auto [Converter, u_inds] = itensor::combiner(inds, {"Tags", "Proj"});
        itensor::ITensor z_tensor(u_inds, itensor::prime(u_inds));
        std::normal_distribution<> dist(0.0, 1.0);
        auto normal_rand = [&dist, &engine]() { return std::complex<double>(dist(engine), dist(engine)); };
        z_tensor.generate(normal_rand);
        auto [Q, R] = itensor::qr(z_tensor, u_inds);
        auto qr_index = itensor::commonIndex(Q, R);
        itensor::ITensor Lambda(qr_index, itensor::prime(u_inds));
        for (int n = 1; n <= itensor::dim(qr_index); ++n) {
            std::complex<double> r = itensor::eltC(R, n, n);
            Lambda.set(n, n, r / std::abs(r));
        }
        Q *= Lambda;
        Q *= Converter;

        return Q;
    }

    /**
    @class tnmc_tebd
    @brief Class managing the TEBD-based tensor network Monte Carlo
    */
    class tnmc_random {
        private:
            tensor_storage::tensor_storage ket_storage_, bra_storage_;
            std::vector<itensor::ITensor> overlap_left_block_bra_, overlap_right_block_bra_, overlap_right_block_ket_, overlap_left_block_ket_;
            int N_, depth_;
            std::mt19937_64 engine_;
            std::uniform_real_distribution<> dist_;
            double log_weight_;
            double overlap_ket_, overlap_bra_;
            std::function<double(double, double)> log_weight_calc_;
            void fill_unitary_(std::vector<std::vector<std::pair<int, itensor::ITensor>>> &gates, int maxdim);
            void update_overlap_local_(int n, direction dir);
            void update_(tensor_storage::tensor_storage &storage, int n, int t);
        public:
            /**
            @brief Constructor for the class that manages the tensor network Monte Calro approach for unitary evolution
            @param [in] gates :  vector of vector of pairs <int, itensor::ITensor> that represents the input unitary gates. A pair <int, itensor::ITensor> contains the left site index to be operated and the unitary gate. The vector of pairs represents one layer. The whole circuit is given as the vector of layers.
            @param [in] psi: An initial matrix-product state used in the unitary evolution
            @param [in] maxdim: The maximum bond dimension used in the tensor network Monte Carlo approach
            @return An instance of the class that manages the tensor network Monte Carlo approach
            @param [inout] std::mt19937_64& engine :  random number generator used for constructing random projectors
            @details Constructor for the class that manages the tensor network Monte Carlo approach. After an initialization, one should call .set_log_weight_function method to set a weight function. Then, .MH_update method is called without observing expectation values enough times for a thermalization. In a measuring phase, .MH_update and .expectation_values methods are called alternately, and one collects the samples of the expectation values.
            */
            tnmc_random(std::vector<std::vector<std::pair<int, itensor::ITensor>>> &gates, itensor::MPS &psi, int maxdim, std::mt19937_64 engine);
            std::vector<double> MH_update();
            /**
            @brief Set weight function used for weighting samples
            @param [in] w_f : function that returns the logarithm of weight computed from the present overlap between bra and ket. The first argument of the function takes the scaled overlap (complex value) and the second argument takes the logarithm of the scale factor.
            @details Weight function introduces the weight for each sample to supress samples that have extraordinary large overlap to make sampling efficient. Initially, the weight function is set to return unity for any input, i.e., no weighting. The weight function cannot be modified after the method \a MH_update is called.
            */
            void set_log_weight_function(std::function<double( double, double)> w_f) {
                log_weight_calc_ = w_f;
                log_weight_ = log_weight_calc_(overlap_bra_, overlap_ket_);
            }
            double log_weight() { return log_weight_; }
            /**
            @brief Expectation values calculated with the present bra and ket
            @param [in] ops : std::vector of itensor::MPO corresponding to observables one wants to evaluate
            @return std::vector of expectation values of input MPOs divided by the weight. The first element is the overlap <bra|ket> / weight. Expectation values are contained from the second element.
            */
            std::vector<std::complex<double>> expectation_values(std::vector<itensor::MPO> &ops);
    };

    inline tnmc_random::tnmc_random(std::vector<std::vector<std::pair<int, itensor::ITensor>>> &gates, itensor::MPS &psi, int maxdim, std::mt19937_64 engine)
        : N_(itensor::length(psi)), depth_(gates.size()), engine_(engine), dist_(0.0, 1.0) {
        overlap_left_block_ket_ = std::vector<itensor::ITensor>(N_);
        overlap_right_block_ket_ = std::vector<itensor::ITensor>(N_);
        overlap_left_block_bra_ = std::vector<itensor::ITensor>(N_);
        overlap_right_block_bra_ = std::vector<itensor::ITensor>(N_);
        log_weight_calc_ = [](double ket, double bra) -> double{ return 0.0; };

        psi.position(1);
        ket_storage_ = tensor_storage::tensor_storage(psi, depth_);

        fill_unitary_(gates, maxdim);

        bra_storage_ = ket_storage_;

        for (int i = 0; i < N_-1; ++i) {
            auto overlap_ket = ket_storage_.get_psi(i, depth_);
            auto overlap_bra = bra_storage_.get_psi(i, depth_);
            auto& ket_psi = ket_storage_.get_psi_ref(i, depth_);
            auto& bra_psi = bra_storage_.get_psi_ref(i, depth_);
            if (i > 0) {
                overlap_ket *= overlap_left_block_ket_.at(i-1);
                overlap_bra *= overlap_left_block_bra_.at(i-1);
            }
            overlap_ket *= itensor::prime(itensor::dag(ket_psi), "Link");
            overlap_bra *= itensor::prime(itensor::dag(bra_psi), "Link");
            overlap_left_block_ket_.at(i) = overlap_ket;
            overlap_left_block_bra_.at(i) = overlap_bra;
        }

        for (int i = N_-1; i > 0; --i) {
            auto overlap_ket = ket_storage_.get_psi(i, depth_);
            auto overlap_bra = bra_storage_.get_psi(i, depth_);
            auto& ket_psi = ket_storage_.get_psi_ref(i, depth_);
            auto& bra_psi = bra_storage_.get_psi_ref(i, depth_);
            if (i < N_-1) {
                overlap_ket *= overlap_right_block_ket_.at(i+1);
                overlap_bra *= overlap_right_block_bra_.at(i+1);
            }
            overlap_ket *= itensor::prime(itensor::dag(ket_psi), "Link");
            overlap_bra *= itensor::prime(itensor::dag(bra_psi), "Link");
            overlap_right_block_ket_.at(i) = overlap_ket;
            overlap_right_block_bra_.at(i) = overlap_bra;
        }

        overlap_ket_ = itensor::eltC(overlap_left_block_ket_.at(N_-2) * overlap_right_block_ket_.at(N_-1)).real();
        overlap_bra_ = itensor::eltC(overlap_left_block_bra_.at(N_-2) * overlap_right_block_bra_.at(N_-1)).real();
        log_weight_ = log_weight_calc_(overlap_bra_, overlap_ket_);
    }


    inline void tnmc_random::fill_unitary_(std::vector<std::vector<std::pair<int, itensor::ITensor>>> &gates, int maxdim) {
        std::vector<bool> is_bond_truncated(N_+1, false);

        for (int t = 0; t < depth_; ++t) {
            std::vector<int> updated_indices{};
            updated_indices.reserve(N_);
            for (auto& x : gates.at(t)) {
                int idx = x.first;
                itensor::ITensor gate = x.second;
                itensor::ITensor Left = ket_storage_.get_psi(idx-1, t);
                itensor::ITensor Right = ket_storage_.get_psi(idx, t);
                itensor::Index left_ind = itensor::uniqueIndex(Left, Right, "Link");
                itensor::Index mid_ind = itensor::commonIndex(Left, Right);
                itensor::Index right_ind = itensor::uniqueIndex(Right, Left, "Link");
                auto left_site = itensor::findIndex(Left, "Site");
                auto right_site = itensor::findIndex(Right, "Site");
                auto [left_gate, right_gate] = itensor::factor(gate, {left_site, itensor::prime(left_site)}, {"Cutoff", 1e-32, "Tags", "Gate"});
                ket_storage_.set_gate(left_gate, idx-1, t);
                ket_storage_.set_gate(right_gate, idx, t);
                itensor::Index gate_ind = itensor::commonIndex(left_gate, right_gate);
                Left *= left_gate;
                Left.noPrime("Site");
                Right *= right_gate;
                Right.noPrime("Site");
                int chi_left = itensor::dim(left_ind), chi_mid = itensor::dim(mid_ind), chi_right = itensor::dim(right_ind);
                itensor::ITensor ProjUnitary;
                int dim = maxdim;
                if (chi_left * itensor::dim(left_site) <= maxdim and !is_bond_truncated.at(idx-1)) {
                    auto [U, S, V] = itensor::svd(Left, {left_ind, left_site}, {"Cutoff", 1e-32, "RightTags", "Proj"});
                    ProjUnitary = itensor::dag(V);
                    dim = itensor::dim(itensor::commonIndex(S, V));
                } else if (chi_right * itensor::dim(right_site) <= maxdim and !is_bond_truncated.at(idx+1)) {
                    auto [U, S, V] = itensor::svd(Right, {mid_ind, gate_ind}, {"Cutoff", 1e-32, "LeftTags", "Proj"});
                    ProjUnitary = U;
                    dim = itensor::dim(itensor::commonIndex(U, S));
                } else {
                    ProjUnitary = random_unitary({mid_ind, gate_ind}, engine_);
                    if (itensor::dim(mid_ind) * itensor::dim(gate_ind) < maxdim) {
                        dim = itensor::dim(mid_ind) * itensor::dim(gate_ind);
                    }
                    ProjUnitary *= std::sqrt(static_cast<double>(itensor::dim(mid_ind) * itensor::dim(gate_ind)) / dim);

                    if (!is_bond_truncated.at(idx)) {
                        is_bond_truncated.at(idx) = true;
                    }
                }
                int proj_n = ket_storage_.calc_projector_index(idx-1, t);
                auto spectrum = tnmc_projector::projector(ProjUnitary, dim);
                ket_storage_.set_projector(spectrum, proj_n);
                itensor::Index proj_idx = itensor::findIndex(ProjUnitary, "Proj");
                itensor::Index proj_sel_idx = itensor::Index(dim, "Proj");
                itensor::ITensor selector(proj_idx, proj_sel_idx);
                std::vector<int> selected;
                selected.reserve(dim);
                for (int k = 1; k <= dim; ++k) {
                    selected.push_back(k);
                    selector.set(k, k, 1.0);
                }
                ProjUnitary *= selector;
                Left *= ProjUnitary;
                Right *= itensor::dag(ProjUnitary);
                Left.replaceTags("Proj", tinyformat::format("Link,l=%d", idx));
                Right.replaceTags("Proj", tinyformat::format("Link,l=%d", idx));
                ket_storage_.set_psi(Left, idx-1, t+1);
                ket_storage_.set_psi(Right, idx, t+1);
                ket_storage_.set_truncated(ProjUnitary, idx-1, t);
                ket_storage_.set_truncated(itensor::dag(ProjUnitary), idx, t);
                ket_storage_.set_selected_modes(selected, proj_n);
                updated_indices.push_back(idx);
                updated_indices.push_back(idx+1);
            }

            for (int idx = 1; idx <= N_; ++idx) {
                auto itr = std::find(updated_indices.begin(), updated_indices.end(), idx);
                if (itr == updated_indices.end()) {
                    ket_storage_.set_psi(ket_storage_.get_psi(idx-1, t), idx-1, t+1);
                }
            }
        }
    }

    inline void tnmc_random::update_(tensor_storage::tensor_storage &storage, int n, int t_ini) {
        for (int t = t_ini; t < depth_; ++t) {
            auto& gate = storage.get_gate_ref(n, t);
            auto psi = storage.get_psi(n, t);
            if ( gate ) {
                auto left_idx = itensor::findIndex(psi, tinyformat::format("Link,l=%d", n));
                auto right_idx = itensor::findIndex(psi, tinyformat::format("Link,l=%d", n+1));
                psi *= gate;
                psi.noPrime("Site");
                auto& truncated_proj = storage.get_truncated_ref(n, t);
                auto proj_link = itensor::findIndex(truncated_proj, "Link");
                psi *= itensor::delta(itensor::findIndex(psi, proj_link.tags()), proj_link);
                psi *= truncated_proj;
                int proj_n;
                if (n == 0) {
                    proj_n = 1;
                } else if (n == N_-1) {
                    proj_n = N_-1;
                }
                else if (itensor::hasIndex(psi, left_idx)) {
                    proj_n = n+1;
                } else {
                    proj_n = n;
                }
                psi.replaceTags("Proj", tinyformat::format("Link,l=%d", proj_n));
            }
            storage.set_psi(psi, n, t+1);
        }
    }

    inline void tnmc_random::update_overlap_local_(int n, direction dir) {
        int left_idx = (dir == direction::to_right ? n : n-1);
        int right_idx = left_idx + 1;
        auto overlap_left_ket = ket_storage_.get_psi(left_idx, depth_);
        auto overlap_right_ket = ket_storage_.get_psi(right_idx, depth_);
        auto overlap_left_bra = bra_storage_.get_psi(left_idx, depth_);
        auto overlap_right_bra = bra_storage_.get_psi(right_idx, depth_);
        auto& ket_left = ket_storage_.get_psi_ref(left_idx, depth_);
        auto& ket_right = ket_storage_.get_psi_ref(right_idx, depth_);
        auto& bra_left = bra_storage_.get_psi_ref(left_idx, depth_);
        auto& bra_right = bra_storage_.get_psi_ref(right_idx, depth_);

        if (left_idx > 0) {
            overlap_left_ket *= overlap_left_block_ket_.at(left_idx-1);
            overlap_left_bra *= overlap_left_block_bra_.at(left_idx-1);
        }

        overlap_left_ket *= itensor::dag(itensor::prime(ket_left, "Link"));
        overlap_left_bra *= itensor::dag(itensor::prime(bra_left, "Link"));

        if (right_idx < N_-1) {
            overlap_right_ket *= overlap_right_block_ket_.at(right_idx+1);
            overlap_right_bra *= overlap_right_block_bra_.at(right_idx+1);
        }
        overlap_right_ket *= itensor::dag(itensor::prime(ket_right, "Link"));
        overlap_right_bra *= itensor::dag(itensor::prime(bra_right, "Link"));

        if (dir == direction::to_right) {
            overlap_left_block_ket_.at(left_idx) = overlap_left_ket;
            overlap_left_block_bra_.at(left_idx) = overlap_left_bra;
        } else {
            overlap_right_block_ket_.at(right_idx) = overlap_right_ket;
            overlap_right_block_bra_.at(right_idx) = overlap_right_bra;
        }

        overlap_ket_ = itensor::eltC(overlap_left_ket * overlap_right_ket).real();
        overlap_bra_ = itensor::eltC(overlap_left_bra * overlap_right_bra).real();
        log_weight_ = log_weight_calc_(overlap_bra_, overlap_ket_);
    }

    inline std::vector<double> tnmc_random::MH_update() {
        int n_try = 0, n_accept = 0, n_skip = 0;

        for (int n = 0; n < (N_-1); ++n) {
            update_overlap_local_(n, direction::to_right);
            for (int t = 0; t < depth_; ++t) {
                int index = bra_storage_.calc_projector_index(n, t);
                double log_w_old = log_weight();
                auto selected_old = bra_storage_.get_selected_modes(index);
                int rank = bra_storage_.projector_rank(index);
                int n_val = bra_storage_.projector_size(index);
                if (rank == n_val) {
                    continue;
                }
                std::vector<int> selected_new;
                selected_new.reserve(rank);
                std::vector<int> indices(n_val);
                std::iota(indices.begin(), indices.end(), 1);
                std::sample(indices.begin(), indices.end(), std::back_inserter(selected_new), rank, engine_);

                if (selected_new == selected_old) {
                    ++n_skip;
                    ++n_try;
                    ++n_accept;
                    continue;
                }

                bra_storage_.reselect_modes(selected_new, n, t);
                update_(bra_storage_, n, t);
                update_(bra_storage_, n+1, t);
                update_overlap_local_(n, direction::to_right);
                double log_w_new = log_weight();

                double threshold = dist_(engine_);
                if (threshold < std::exp(log_w_new - log_w_old)) {
                    ++n_accept;
                } else {
                    bra_storage_.reselect_modes(selected_old, n, t);
                    update_(bra_storage_, n, t);
                    update_(bra_storage_, n+1, t);
                    update_overlap_local_(n, direction::to_right);
                }
                ++n_try;
            }
        }

        for (int n = (N_-2); n >= 0; --n) {
            update_overlap_local_(n+1, direction::to_left);
            for (int t = 0; t < depth_; ++t) {
                int index = ket_storage_.calc_projector_index(n, t);
                double log_w_old = log_weight();
                auto selected_old = ket_storage_.get_selected_modes(index);
                int rank = ket_storage_.projector_rank(index);
                int n_val = ket_storage_.projector_size(index);
                if (rank == n_val) {
                    continue;
                }
                std::vector<int> selected_new;
                selected_new.reserve(rank);
                std::vector<int> indices(n_val);
                std::iota(indices.begin(), indices.end(), 1);
                std::sample(indices.begin(), indices.end(), std::back_inserter(selected_new), rank, engine_);


                if (selected_new == selected_old) {
                    ++n_skip;
                    ++n_try;
                    ++n_accept;
                    continue;
                }

                ket_storage_.reselect_modes(selected_new, n, t);
                update_(ket_storage_, n, t);
                update_(ket_storage_, n+1, t);
                update_overlap_local_(n+1, direction::to_left);
                double log_w_new = log_weight();

                double threshold = dist_(engine_);
                if (threshold < std::exp(log_w_new - log_w_old)) {
                    ++n_accept;
                } else {
                    ket_storage_.reselect_modes(selected_old, n, t);
                    update_(ket_storage_, n+1, t);
                    update_(ket_storage_, n, t);
                    update_overlap_local_(n+1, direction::to_left);
                }
                ++n_try;
            }
        }
        return {static_cast<double>(n_accept) / n_try, static_cast<double>(n_skip) / n_try};
    }

    inline std::vector<std::complex<double>> tnmc_random::expectation_values(std::vector<itensor::MPO> &ops) {
        std::vector<std::complex<double>> results;
        itensor::MPS ket(N_), bra(N_);
        for (int i=0; i < N_; ++i) {
            auto& psi = ket_storage_.get_psi_ref(i, depth_);
            ket.Aref(i+1) = psi;
        }
        for (int i=0; i < N_; ++i) {
            auto& psi = bra_storage_.get_psi_ref(i, depth_);
            bra.Aref(i+1) = psi;
        }
        results.reserve(ops.size()+1);
        double log_w = log_weight();
        results.push_back(itensor::innerC(bra, ket)*std::exp(-log_w));
        for (auto &x : ops) {
            results.push_back(itensor::innerC(bra, x, ket)*std::exp(-log_w));
        }
        return results;
    }
}
