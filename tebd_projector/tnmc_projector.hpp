/**
 @file tnmc_projector.hpp
 @brief Header file for the class manages a projector
*/
#pragma once
#include <itensor/all_basic.h>
#include <vector>

namespace tensor_storage {
    class tensor_storage;
}

/**
 @namespace tnmc_projector
 @brief Namespace that contains the class for a projector used for tensor network Monte Carlo
 */
namespace tnmc_projector {
    /**
    @class projector
    @brief Class that manages a projector obtained by singular value decomposition
    */
    class projector {
        private:
            itensor::ITensor left_projector_;
            itensor::ITensor right_projector_;
            std::vector<double> probabilities_, singular_values_, weight_;
            std::vector<std::vector<double>> S_param_, T_param_;
            int rank_, mode_n_, det_n_;
            double random_prob_;
            void update_S_T_params_(const std::vector<double> &weight);
            friend class tensor_storage::tensor_storage;
        public:
            /**
              @brief Constructor for a projector
              @param [in] left : Left side projector
              @param [in] right : Right side projector
              @param [in] rank_in : Rank of a truncated projector
              @param [in] svals : Singular values obtained during singular value decomposition
              @param [in, opt] random_prob : The target probability for a random mode. The default value is set to 1.0e-6.
              @param [in, opt] power : The power for modifying input singular values. The default value is 1.0, i.e., no modification.
              @param [in, opt] rel_error : Relative error threshold used in the bisection method. The default value is 1.0e-2.
              @param [in, opt] max_iter : Maximum iteration used in the bisection method. The default value is 1000.
              @details Constructor for a projector. Singular values for random modes are determined so that the probability for a random mode becomes the preset target value by the bisection method.
              */
            projector(const itensor::ITensor &left, const itensor::ITensor &right, int rank_in, int n_random, std::vector<double> &svals, double random_prob = 1e-6, double power=1, double rel_error=1e-2, int max_iter=1000);
            /**
              @brief Default constructor for an empty projector
              */
            projector();
            /**
             @brief Get the preset rank for the projector 
             @return The preset rank for the projector
            */
            int rank() { return rank_; }
            /**
             @brief Get the number of modes in the rank full projector
             @return The number of modes in the rank full projector
            */
            int mode_n() { return mode_n_; }
            /**
             @brief Get the all of the probabilities for selecting the modes
             @return The vector contains all the probability for selecting modes
            */
            std::vector<double> probabilities() { return probabilities_; }
            /**
             @brief Get the conditional probability for selecting the \a k-th mode when m modes are already selected
             @param [in] k : Index to specify a mode (0-indexed)
             @param [in] m : Number of already selected modes
             @return The probability for selecting the \a k-th mode when m modes are already selected
            */
            double get_conditional_probability (int k, int m);
            /**
             @brief Get the probability for selecting the specified mode
             @param [in] k : Index to specify a mode (0-indexed)
             @return The probability for selecting the mode \a k
            */
            double get_marginal_probability (int k) { return probabilities_.at(k); };
    };

    inline projector::projector(const itensor::ITensor &left, const itensor::ITensor &right, int rank_in, int n_random, std::vector<double> &svals, double random_prob, double power, double rel_err, int max_iter) : left_projector_(left), right_projector_(right), singular_values_(svals), rank_(rank_in), mode_n_(svals.size() + n_random), random_prob_(random_prob) {
        weight_.reserve(mode_n_);
        for (auto &x : singular_values_) {
            weight_.push_back(std::pow(x, power));
        }
        for (int i = 0; i < n_random; ++i) {
            weight_.push_back(0.0);
        }
        det_n_ = singular_values_.size();

        S_param_.resize(mode_n_+1);
        for (int i = 0; i < mode_n_; ++i) {
            S_param_.at(i).resize(rank_+1);
            S_param_.at(i).at(rank_) = 1.0;
        }
        std::vector<double> s_r(rank_+1, 0.0);
        s_r.at(rank_) = 1.0;
        S_param_.at(mode_n_) = s_r;

        probabilities_.resize(mode_n_+1);
        T_param_.resize(mode_n_+1);
        std::vector<double> t_0(rank_+1, 0.0);
        t_0.at(0) = 1.0;
        T_param_.at(0) = t_0;

        if (n_random > 0) {
            double exp_low = -16.0, exp_high = 0.0;
            std::vector<double> weight_tmp = weight_;

            while (true) {
                double random_weight = std::pow(10.0, exp_low) * weight_.at(0);
                for (int i = 0; i < n_random; ++i) {
                    weight_tmp.at(det_n_+i) = random_weight;
                }
                update_S_T_params_(weight_tmp);
                if (n_random * probabilities_.at(det_n_+1) < random_prob_) {
                    break;
                }
                exp_low = exp_low - 1.0;
            }

            while (true) {
                double random_weight = std::pow(10.0, exp_high) * weight_.at(0);
                for (int i = 0; i < n_random; ++i) {
                    weight_tmp.at(det_n_+i) = random_weight;
                }
                update_S_T_params_(weight_tmp);
                if (n_random * probabilities_.at(det_n_+1) > random_prob_) {
                    break;
                }
                exp_high = exp_high + 1.0;
            }

            for (int i = 0; i < max_iter; ++i) {
                double exp_mid = 0.5*(exp_high + exp_low);
                double random_weight = std::pow(10.0, exp_mid) * weight_.at(0);
                for (int i = 0; i < n_random; ++i) {
                    weight_tmp.at(det_n_+i) = random_weight;
                }
                update_S_T_params_(weight_tmp);
                double p_mid = probabilities_.at(det_n_+1);
                if (std::abs((p_mid - random_prob_)/p_mid) < rel_err) {
                    if (random_weight > weight_.at(det_n_-1)) {
                        for (int i = 0; i < n_random; ++i) {
                            weight_tmp.at(det_n_+i) = 0.5*weight_.at(det_n_-1);
                        }
                        update_S_T_params_(weight_tmp);
                    }
                    weight_ = weight_tmp;
                    break;
                }
                if (p_mid > random_prob_) {
                    exp_high = exp_mid;
                } else {
                    exp_low = exp_mid;
                }

                if (i == max_iter-1) {
                    std::cout << "Not converged, achieved probability: " << p_mid << std::endl; 
                    weight_ = weight_tmp;
                }
            }
        } else {
            update_S_T_params_(weight_);
        }

        auto proj_idx = itensor::findIndex(left_projector_, "Proj");
        itensor::ITensor left_weight(proj_idx, itensor::prime(proj_idx)),
                         right_weight(proj_idx, itensor::prime(proj_idx));
        for (int i = 1; i <= det_n_; ++i) {
            double p = probabilities_.at(i);
            left_weight.set(i, i, 1.0/p);
            right_weight.set(i, i, 1.0);
        }
        for (int i = det_n_+1; i <= mode_n_; ++i) {
            double sqrt_p = std::sqrt(probabilities_.at(i));
            left_weight.set(i, i, 1.0/sqrt_p);
            right_weight.set(i, i, 1.0/sqrt_p);
        }
        left_projector_ *= left_weight;
        left_projector_.noPrime("Proj");
        right_projector_ *= right_weight;
        right_projector_.noPrime("Proj");
    }

    inline void projector::update_S_T_params_(const std::vector<double> &weight) {
        for (int k = mode_n_-1; k >= 0; --k) {
            S_param_.at(k).at(rank_-1) = weight.at(k) + S_param_.at(k+1).at(rank_-1);
        }
        for (int k = mode_n_-1; k >= 0; --k) {
            for (int m = rank_-2; m >= 0; --m) {
                S_param_.at(k).at(m) = S_param_.at(k+1).at(m+1)
                    *(weight.at(k)+S_param_.at(k+1).at(m))
                    /(weight.at(k)+S_param_.at(k+1).at(m+1));
            }
        }

        for (int k=1; k <= mode_n_; ++k) {
            std::vector<double> t_k(rank_+1);
            t_k.at(0) = 1.0;
            double w = weight.at(k-1);
            t_k.at(1) = w + T_param_.at(k-1).at(1);
            for (int m=2; m <= rank_; ++m) {
                t_k.at(m) = T_param_.at(k-1).at(m-1)
                    * (w + T_param_.at(k-1).at(m))
                    / (w + T_param_.at(k-1).at(m-1));
            }
            T_param_.at(k) = t_k;
        }

        double factor = 1.0 / std::sqrt(S_param_.at(0).at(0) * T_param_.at(mode_n_).at(0));
        for (int k=1; k <= mode_n_; ++k) {
            double sum = 0.0;
            for (int m=0; m < rank_; ++m) {
                double first_fac = 1.0, second_fac = 1.0;
                for (int j=1; j<=m; ++j) {
                    first_fac *= T_param_.at(k-1).at(j) / std::sqrt(S_param_.at(0).at(j)*T_param_.at(mode_n_).at(j));
                }
                for (int j=m+1; j <= rank_; ++j) {
                    second_fac *= S_param_.at(k).at(j) / std::sqrt(S_param_.at(0).at(j)*T_param_.at(mode_n_).at(j));
                }
                sum += first_fac * second_fac;
            }
            probabilities_.at(k) = weight.at(k-1) * sum * factor;
        }
    }

    inline double projector::get_conditional_probability (int k, const int m) {
        if (k == 0 or k > weight_.size()) {
            return 0.0;
        }
        double w = weight_.at(k-1);
        double s_param = S_param_.at(k).at(m);
        return w / (w + s_param);
    }
    /**
      @brief Default constructor for an empty projector
      @details This constructor requires for initializing STL continers.
      */
    inline projector::projector() {
        left_projector_ = itensor::ITensor();
        right_projector_ = itensor::ITensor();
        probabilities_ = {};
        S_param_ = std::vector<std::vector<double>>();
        T_param_ = std::vector<std::vector<double>>();
    }
}
