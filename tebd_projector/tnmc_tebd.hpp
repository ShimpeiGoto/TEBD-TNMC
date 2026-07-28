/**
@file tnmc_tebd.hpp
@brief Header file for TEBD-based tensor network Monte Carlo
*/
#pragma once
#include <algorithm>
#include <vector>
#include <random>
#include <itensor/all_mps.h>
#include "itensor/mps/mpo.h"
#include "tnmc_projector.hpp"
#include "tensor_storage.hpp"
#include "svd_projector.hpp"

/**
@namespace tnmc_tebd
@brief Namespace containing the class that manages TEBD-based tensor network Monte Carlo
*/
namespace tnmc_tebd {
    /**
    @brief Enumerator representing the direction of local overlap update
    */
    enum class direction {
        to_left, ///< Direction to left
        to_right ///< Direction to right
    };

    /**
    @class tnmc_tebd
    @brief Class managing the TEBD-based tensor network Monte Carlo
    */
    class tnmc_tebd {
        private:
            tensor_storage::tensor_storage ket_storage_, bra_storage_;
            std::vector<itensor::ITensor> overlap_left_block_bra_, overlap_right_block_bra_, overlap_right_block_ket_, overlap_left_block_ket_;
            int N_, depth_;
            double log_weight_, pinv_threshold_, random_prob_, singular_value_power_;
            double log_overlap_ket_, log_overlap_bra_;
            std::uniform_real_distribution<double> dist_;
            std::function<double(double, double)> log_weight_calc_;
            void initial_tebd_(std::vector<std::vector<std::pair<int, itensor::ITensor>>> &gates, int maxdim);
            void update_overlap_local_(int n, direction dir);
            void update_(tensor_storage::tensor_storage &storage, int n, int t);
        public:

            /**
            @brief Constructor for the class that manages the tensor network Monte Calro approach for unitary evolution
            @param [in] gates :  vector of vector of pairs <int, itensor::ITensor> that represents the input unitary gates. A pair <int, itensor::ITensor> contains the left site index to be operated and the unitary gate. The vector of pairs represents one layer. The whole circuit is given as the vector of layers.
            @param [in] psi: An initial matrix-product state used in the unitary evolution
            @param [in] maxdim: The maximum bond dimension used in the tensor network Monte Carlo approach
            @param [in,opt] pinv_threshold: Singular value threshold used for computing pseudo-inverse matrix. The default value is set to 1e-6.
            @param [in,opt] random_prob: Target probability for selecting a random mode. The default value is set to 1e-6.
            @param [in,opt] singular_value_power: Modification power for singular values in calculating probabilities for selecting singular modes. The default value is set to 1.0 that corresponds to no modification.
            @return An instance of the class that manages the tensor network Monte Carlo approach
            @details Constructor for the class that manages the tensor network Monte Carlo approach. After an initialization, one should call .set_log_weight_function method to set a weight function. Then, .MH_update method is called without observing expectation values enough times for a thermalization. In a measuring phase, .MH_update and .expectation_values methods are called alternately, and one collects the samples of the expectation values.
            */
            tnmc_tebd(std::vector<std::vector<std::pair<int, itensor::ITensor>>> &gates, itensor::MPS &psi, int maxdim, double pinv_threshold=1e-6, double random_prob=1e-6, double singular_value_power=1.0);
            /**
            @brief Perform updates of projectors and accepts / rejects the updates based on the Metropolis-Hasting algorithm
            @param [inout] std::mt19937_64& engine :  random number generator used in samplings
            @return Three-dimensional std::vector containing statistical information of samplings. The vector is composed of (acceptance rate, skip rate, random modes rate)
            @details The function that performs the updates of projectors and that accepts / rejects the updates based on the Metropolis-Hasting algorithm. The expectation values should be observed after this function is called.
            */
            std::vector<double> MH_update(std::mt19937_64 &engine);
            /**
            @brief Set weight function used for weighting samples
            @param [in] w_f : function that returns the logarithm of weight computed from the present overlap between bra and ket. The first (second) argument of the function takes the logarithm of the overlap of the bra (ket) part. 
            @details Weight function for weighting sampling. To avoid numeircal overflow, the input and the output of the function take logarithms.
            */
            void set_log_weight_function(std::function<double(double, double)> w_f) {
                log_weight_calc_ = w_f;
                log_weight_ = log_weight_calc_(log_overlap_bra_, log_overlap_ket_);
            }
            /**
             @brief Get the logarithm of weight
             @return Logarithm of weight
             @details Get the logarithm of weight
             */
            double log_weight() { return log_weight_; }
            /**
             @brief Get the logarithm of the overlap of the ket part
             @return Logarithm of the overlap of the ket part
             @details Get the logarithm of the overlap of the ket part
             */
            double log_overlap_ket() { return log_overlap_ket_; }
            /**
             @brief Get the logarithm of the overlap of the bra part
             @return Logarithm of the overlap of the bra part
             @details Get the logarithm of the overlap of the bra part
             */
            double log_overlap_bra() { return log_overlap_bra_; }
            /**
            @brief Expectation values calculated with the present bra and ket
            @param [in] ops : std::vector of itensor::MPO corresponding to observables one wants to evaluate
            @return std::vector of expectation values of input MPOs divided by the weight. The first element is the overlap <bra|ket> / weight. Expectation values are contained from the second element.
            */
            std::vector<std::complex<double>> expectation_values(std::vector<itensor::MPO> &ops);
    };

    inline tnmc_tebd::tnmc_tebd(std::vector<std::vector<std::pair<int, itensor::ITensor>>> &gates, itensor::MPS &psi, int maxdim, double pinv_threshold, double random_prob, double singular_value_power)
        : N_(itensor::length(psi)), depth_(gates.size()), pinv_threshold_(pinv_threshold), random_prob_(random_prob), singular_value_power_(singular_value_power), dist_(0.0, 1.0) {
        overlap_left_block_ket_ = std::vector<itensor::ITensor>(N_);
        overlap_right_block_ket_ = std::vector<itensor::ITensor>(N_);
        overlap_left_block_bra_ = std::vector<itensor::ITensor>(N_);
        overlap_right_block_bra_ = std::vector<itensor::ITensor>(N_);
        log_weight_calc_ = [](double ket, double bra) -> double{ return 0.0; };

        psi.position(1);
        ket_storage_ = tensor_storage::tensor_storage(psi, depth_);

        initial_tebd_(gates, maxdim);

        bra_storage_ = ket_storage_;

        for (int i = 0; i < N_; ++i) {
            update_(ket_storage_, i, 0);
            update_(bra_storage_, i, 0);
        }

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

        log_overlap_ket_ = std::log(itensor::eltC(overlap_left_block_ket_.at(N_-2) * overlap_right_block_ket_.at(N_-1)).real()) + 2.0*ket_storage_.log_norm_total();
        log_overlap_bra_ = std::log(itensor::eltC(overlap_left_block_bra_.at(N_-2) * overlap_right_block_bra_.at(N_-1)).real()) + 2.0*bra_storage_.log_norm_total();
        log_weight_ = log_weight_calc_(log_overlap_bra_, log_overlap_ket_);
    }


    inline void tnmc_tebd::initial_tebd_(std::vector<std::vector<std::pair<int, itensor::ITensor>>> &gates, int maxdim) {
        std::vector<std::vector<double>> Lambdas(N_-1);
        for (int i = 1; i < N_; ++i) {
            itensor::ITensor M = ket_storage_.get_psi(i-1, 0);
            auto l = itensor::findIndex(M, tinyformat::format("Link,l=%d",i-1));
            auto s = itensor::findIndex(M, "Site");
            auto [U, S, V] = itensor::svd(M, {l, s}, {"Truncate", false, "RightTags", tinyformat::format("Link,l=%d",i)});
            int rank = itensor::dim(itensor::commonIndex(U, S));
            std::vector<double> svals;
            svals.reserve(rank);
            for (int n = 1; n <= rank; ++n) {
                svals.push_back(itensor::elt(S, n, n));
            }
            Lambdas.at(i-1) = svals;
            if (i > 1) {
                auto svals_prev = Lambdas.at(i-2);
                itensor::ITensor inv_Lambda(l, itensor::prime(l));
                int rank_prev = svals_prev.size();
                for (int k = 1; k <= rank_prev; ++k) {
                    inv_Lambda.set(k, k, 1.0/svals_prev.at(k-1));
                }
                U *= inv_Lambda;
                U.noPrime();
            }
            ket_storage_.set_psi(U*S, i-1, 0);
            auto N = ket_storage_.get_psi(i, 0);
            ket_storage_.set_psi(V*N, i, 0);
        }

        for (int t = 0; t < depth_; ++t) {
            std::vector<int> updated_indices{};
            updated_indices.reserve(N_);
            for (auto& x : gates.at(t)) {
                int idx = x.first;
                itensor::ITensor gate = x.second;
                if (gate.order() == 2) {
                    // single-qubit gate
                    itensor::ITensor A = ket_storage_.get_psi(idx-1, t);
                    A *= gate;
                    A.noPrime("Site");
                    ket_storage_.set_gate(gate, idx-1, t);
                    ket_storage_.set_psi(A, idx-1, t+1);
                    updated_indices.push_back(idx);
                } else {
                    // two-qubit gate
                    itensor::ITensor Left = ket_storage_.get_psi(idx-1, t);
                    itensor::ITensor Right = ket_storage_.get_psi(idx, t);
                    auto left_site = itensor::findIndex(Left, "Site");
                    auto [left_gate, right_gate] = itensor::factor(gate, {left_site, itensor::prime(left_site)}, {"Cutoff", 1e-32, "Tags", "Gate"});
                    ket_storage_.set_gate(left_gate, idx-1, t);
                    ket_storage_.set_gate(right_gate, idx, t);
                    Left *= left_gate;
                    Left.noPrime("Site");
                    Right *= right_gate;
                    Right.noPrime("Site");
                    itensor::ITensor LambdaLeft;
                    if (idx == 1) {
                        LambdaLeft = Left;
                    } else {
                        auto left_link = itensor::findIndex(Left, tinyformat::format("Link,l=%d",idx-1));
                        itensor::ITensor Lambda(left_link, itensor::prime(left_link));
                        auto& lambda_vec = Lambdas.at(idx-2);
                        for (int i = 0; i < lambda_vec.size(); ++i) {
                            Lambda.set(i+1, i+1, lambda_vec.at(i));
                        }
                        LambdaLeft = Lambda * Left;
                        LambdaLeft.noPrime();
                    }
                    auto [LeftProj, RightProj, svals, n_random] = svd_projector::projector(LambdaLeft, Right, "Proj", 0, pinv_threshold_);
                    int rank = svals.size();
                    int dim = rank;
                    if (rank > maxdim) {
                        dim = maxdim;
                    }
                    std::vector<double> lambda_sel(dim);
                    std::copy(svals.begin(), svals.begin()+dim, lambda_sel.begin());
                    Lambdas.at(idx-1) = lambda_sel;
                    int proj_n = ket_storage_.calc_projector_index(idx-1, t);
                    auto spectrum = tnmc_projector::projector(LeftProj, RightProj, dim, n_random, svals, random_prob_, singular_value_power_);
                    ket_storage_.set_projector(spectrum, proj_n);
                    itensor::Index proj_idx = itensor::findIndex(LeftProj, "Proj");
                    itensor::Index proj_sel_idx = itensor::Index(dim, "Proj");
                    itensor::ITensor selector(proj_idx, proj_sel_idx),
                                     selector_left_modified(proj_idx, proj_sel_idx),
                                     selector_right_modified(proj_idx, proj_sel_idx);
                    std::vector<int> selected;
                    for (int k = 1; k <= dim; ++k) {
                        double p = ket_storage_.get_marginal_probability(k, proj_n);
                        selected.push_back(k);
                        selector.set(k, k, 1.0);
                        selector_left_modified.set(k, k, 1.0 / p);
                        selector_right_modified.set(k, k, 1.0);
                    }
                    auto LeftProjMod = LeftProj * selector_left_modified;
                    auto RightProjMod = RightProj * selector_right_modified;
                    LeftProj *= selector;
                    RightProj *= selector;
                    LeftProj.replaceTags("Proj", tinyformat::format("Link,l=%d", idx));
                    RightProj.replaceTags("Proj", tinyformat::format("Link,l=%d", idx));
                    Right *= RightProj;
                    Left *= LeftProj;
                    ket_storage_.set_psi(Left, idx-1, t+1);
                    ket_storage_.set_psi(Right, idx, t+1);
                    ket_storage_.set_truncated(LeftProjMod, idx-1, t);
                    ket_storage_.set_truncated(RightProjMod, idx, t);
                    ket_storage_.set_selected_modes(selected, proj_n);
                    updated_indices.push_back(idx);
                    updated_indices.push_back(idx+1);
                }
            }

            for (int idx = 1; idx <= N_; ++idx) {
                auto itr = std::find(updated_indices.begin(), updated_indices.end(), idx);
                if (itr == updated_indices.end()) {
                    ket_storage_.set_psi(ket_storage_.get_psi(idx-1, t), idx-1, t+1);
                }
            }
        }
    }

    inline void tnmc_tebd::update_(tensor_storage::tensor_storage &storage, int n, int t_ini) {
        for (int t = t_ini; t < depth_; ++t) {
            auto& gate = storage.get_gate_ref(n, t);
            auto psi = storage.get_psi(n, t);
            if ( gate ) {
                if (itensor::order(gate) == 2) {
                    psi *= gate;
                    psi.noPrime(("Site"));
                } else {
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
            }
            double log_norm = storage.get_log_norm(n, t);
            double norm = itensor::norm(psi);
            psi /= norm;
            log_norm += std::log(norm);
            storage.set_psi(psi, n, t+1);
            storage.set_log_norm(log_norm, n, t+1);
        }
    }

    inline void tnmc_tebd::update_overlap_local_(int n, direction dir) {
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

        log_overlap_ket_ = std::log(itensor::eltC(overlap_left_ket * overlap_right_ket).real()) + 2.0*ket_storage_.log_norm_total();
        log_overlap_bra_ = std::log(itensor::eltC(overlap_left_bra * overlap_right_bra).real()) + 2.0*bra_storage_.log_norm_total();
        log_weight_ = log_weight_calc_(log_overlap_bra_, log_overlap_ket_);
    }

    inline std::vector<double> tnmc_tebd::MH_update(std::mt19937_64 &engine) {
        int n_try = 0, n_accept = 0, n_skip = 0, n_random = 0;

        for (int n = 0; n < (N_-1); ++n) {
            update_overlap_local_(n, direction::to_right);
            for (int t = 0; t < depth_; ++t) {
                int index = bra_storage_.calc_projector_index(n, t);
                double log_w_old = log_weight();
                auto selected_old = bra_storage_.get_selected_modes(index);
                int rank = bra_storage_.projector_rank(index);
                int n_val = bra_storage_.projector_size(index);
                int n_det = bra_storage_.projector_deterministic_mode_n(index);
                if (rank == n_val) {
                    continue;
                }
                std::vector<int> selected_new;
                selected_new.reserve(rank);

                int n_selected = 0, n_random_new = 0;
                for (int k = 1; k <= n_val; ++k) {
                    double prob = bra_storage_.get_conditional_probability(k, n_selected, index);
                    if (dist_(engine) < prob) {
                        selected_new.push_back(k);
                        ++n_selected;
                        if (k > n_det) {
                            ++n_random_new;
                        }
                    }

                    if (n_selected == rank) {
                        break;
                    }
                }
                
                if (selected_new == selected_old) {
                    ++n_skip;
                    ++n_try;
                    ++n_accept;
                    n_random += n_random_new;
                    continue;
                }

                bra_storage_.reselect_modes(selected_new, n, t);
                update_(bra_storage_, n, t);
                update_(bra_storage_, n+1, t);
                update_overlap_local_(n, direction::to_right);
                double log_w_new = log_weight();

                double threshold = dist_(engine);
                if (threshold < std::exp(log_w_new - log_w_old)) {
                    ++n_accept;
                    n_random += n_random_new;
                } else {
                    bra_storage_.reselect_modes(selected_old, n, t);
                    update_(bra_storage_, n, t);
                    update_(bra_storage_, n+1, t);
                    update_overlap_local_(n, direction::to_right);
                    int n_random_old = 0;
                    for (auto k : selected_old) {
                        if (k > n_det) {
                            ++n_random_old;
                        }
                    }
                    n_random += n_random_old;
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
                int n_det = ket_storage_.projector_deterministic_mode_n(index);
                if (rank == n_val) {
                    continue;
                }
                std::vector<int> selected_new;
                selected_new.reserve(rank);

                int n_selected = 0, n_random_new = 0;
                for (int k = 1; k <= n_val; ++k) {
                    double prob = ket_storage_.get_conditional_probability(k, n_selected, index);
                    if (dist_(engine) < prob) {
                        selected_new.push_back(k);
                        ++n_selected;
                        if (k > n_det) {
                            ++n_random_new;
                        }
                    }

                    if (n_selected == rank) {
                        break;
                    }
                }

                if (selected_new == selected_old) {
                    ++n_skip;
                    ++n_try;
                    ++n_accept;
                    n_random += n_random_new;
                    continue;
                }

                ket_storage_.reselect_modes(selected_new, n, t);
                update_(ket_storage_, n, t);
                update_(ket_storage_, n+1, t);
                update_overlap_local_(n+1, direction::to_left);
                double log_w_new = log_weight();

                double threshold = dist_(engine);
                if (threshold < std::exp(log_w_new - log_w_old)) {
                    n_random += n_random_new;
                    ++n_accept;
                } else {
                    ket_storage_.reselect_modes(selected_old, n, t);
                    update_(ket_storage_, n+1, t);
                    update_(ket_storage_, n, t);
                    update_overlap_local_(n+1, direction::to_left);
                    int n_random_old = 0;
                    for (auto k : selected_old) {
                        if (k > n_det) {
                            ++n_random_old;
                        }
                    }
                    n_random += n_random_old;
                }
                ++n_try;
            }
        }
        return {static_cast<double>(n_accept) / n_try, static_cast<double>(n_skip) / n_try, static_cast<double>(n_random) / n_try};
    }

    inline std::vector<std::complex<double>> tnmc_tebd::expectation_values(std::vector<itensor::MPO> &ops) {
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
        double log_norm = ket_storage_.log_norm_total() + bra_storage_.log_norm_total();
        results.reserve(ops.size()+1);
        double log_w = log_weight();
        results.push_back(itensor::innerC(bra, ket)*std::exp(log_norm-log_w));
        for (auto &x : ops) {
            results.push_back(itensor::innerC(bra, x, ket)*std::exp(log_norm-log_w));
        }
        return results;
    }
}
