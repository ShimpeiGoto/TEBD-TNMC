/**
 @file svd_projector.hpp
 @brief Header file contains routines that generates a SVD-based projector
 */
#pragma once
#include <cmath>
#include <itensor/all_basic.h>

/**
 @namespace svd_projector
 @brief Namespace that contains routines for generating a SVD-based projector
 */
namespace svd_projector {
    /**
     @brief Reorthogonalize the modes containing in ITensors ProjL and ProjR
     @param [in,out] ProjL : ITensor that contains modes to be reorthogonalized against modes in ProjR
     @param [in,out] ProjR : ITensor that contains modes to be reorthogonalized against modes in ProjL
     @param [in] proj_tags : Tags specfying the index that labels the modes in the input ITensors
     */
    inline void reorthogonalize(itensor::ITensor &ProjL, itensor::ITensor &ProjR, const itensor::TagSet &proj_tags) {
        std::vector<itensor::Index> common_inds;
        itensor::Index proj_idx;
        int rank = 0;
        for (auto &x : ProjL.inds()) {
            if (itensor::hasTags(x, proj_tags)) {
                rank = itensor::dim(x);
                proj_idx = x;
            } else {
                common_inds.push_back(x);
            }
        }

        auto [Combiner, combiner_ind] = itensor::combiner(common_inds);
        ProjL *= Combiner;
        ProjR *= Combiner;

        std::vector<itensor::ITensor> lstates, rstates;
        lstates.reserve(rank);
        rstates.reserve(rank);

        ProjL.permute({combiner_ind, proj_idx});
        ProjR.permute({combiner_ind, proj_idx});
        for (int i = 1; i <= rank; ++i) {
            auto lstate = itensor::ITensor(combiner_ind);
            auto rstate = itensor::ITensor(combiner_ind);
            for (int j = 1; j <= itensor::dim(combiner_ind); ++j) {
                lstate.set(j, itensor::eltC(ProjL, j, i));
                rstate.set(j, itensor::eltC(ProjR, j, i));
            }
            lstates.push_back(lstate);
            rstates.push_back(rstate);
        }

        for (int i = 0; i < rank; ++i) {
            for (int j = 0; j < i; ++j) {
                std::complex<double> CoeffL = itensor::eltC(rstates.at(j) * lstates.at(i));
                lstates.at(i) -= CoeffL * lstates.at(j);
                std::complex<double> CoeffR = itensor::eltC(lstates.at(j) * rstates.at(i));
                rstates.at(i) -= CoeffR * rstates.at(j);
            }

            std::complex<double> norm = itensor::eltC(lstates.at(i) * rstates.at(i));
            lstates.at(i) /= (norm/std::sqrt(std::abs(norm)));
            rstates.at(i) /= std::sqrt(std::abs(norm));
        }

        for (int i = 1; i <= rank; ++i) {
            for (int j = 1; j <= itensor::dim(combiner_ind); ++j) {
                ProjR.set(j, i, itensor::eltC(rstates.at(i-1), j));
                ProjL.set(j, i, itensor::eltC(lstates.at(i-1), j));
            }
        }

        ProjL *= Combiner;
        ProjR *= Combiner;
    }

    /**
     @brief Fill truncated modes by random modes orthogonal to existing modes so that ITensors ProjL and ProjR become full rank
     @param [in,out] ProjL : Rank deficient ITensor to be filled
     @param [in,out] ProjR : Rank deficient ITensor to be filled
     @param [in] ker : The dimension of kernel space, i.e., the number of modes to be filled
     @param [in] proj_tags : Tags specfying the index that labels the modes in the input ITensors
     */
    inline void fill_kernel(itensor::ITensor &ProjL, itensor::ITensor &ProjR, int ker, const itensor::TagSet proj_tags) {
        std::vector<itensor::Index> common_inds;
        itensor::Index proj_idx;
        int rank = 0;
        for (auto &x : ProjL.inds()) {
            if (itensor::hasTags(x, proj_tags)) {
                rank = itensor::dim(x);
                proj_idx = x;
            } else {
                common_inds.push_back(x);
            }
        }
        itensor::ITensor Projector = ProjL*itensor::prime(ProjR, common_inds);
        std::vector<itensor::Index> proj_inds;
        proj_inds.reserve(2*common_inds.size());
        for (auto& x : common_inds) {
            proj_inds.push_back(x);
            proj_inds.push_back(itensor::prime(x));
        }
        itensor::ITensor id(proj_inds);
        std::vector<std::vector<int>> id_indices;
        id_indices.reserve(rank);
        for (auto &x : common_inds) {
            if (id_indices.size() == 0) {
                for (int i = 1; i <= itensor::dim(x); ++i) {
                    id_indices.push_back(std::vector<int>{i, i});
                }
            } else {
                std::vector<std::vector<int>> next;
                next.reserve(id_indices.size()*itensor::dim(x));
                for (int i = 1; i <= itensor::dim(x); ++i) {
                    for (auto y : id_indices) {
                        y.push_back(i);
                        y.push_back(i);
                        next.push_back(y);
                    }
                }
                id_indices = next;
            }
        }

        for (auto &x : id_indices) {
            id.set(x, 1.0);
        }

        Projector = id - Projector;
        std::vector<itensor::ITensor> LKernel;
        std::vector<itensor::ITensor> RKernel;
        LKernel.reserve(ker);
        RKernel.reserve(ker);
        for (int i = 0; i < ker; ++i) {
            auto lstate = itensor::randomITensorC(common_inds);
            lstate.prime();
            lstate *= Projector;
            LKernel.push_back(lstate);
            auto rstate = itensor::randomITensorC(common_inds);
            rstate *= Projector;
            RKernel.push_back(itensor::prime(rstate, -1));
        }


        for (int i = 0; i < ker; ++i) {
            for (int j = 0; j < i; ++j) {
                std::complex<double> CoeffL = itensor::eltC(RKernel.at(j) * LKernel.at(i));
                LKernel.at(i) -= CoeffL * LKernel.at(j);
                std::complex<double> CoeffR = itensor::eltC(LKernel.at(j) * RKernel.at(i));
                RKernel.at(i) -= CoeffR * RKernel.at(j);
            }

            std::complex<double> norm = itensor::eltC(LKernel.at(i) * RKernel.at(i));
            LKernel.at(i) /= (norm/std::sqrt(std::abs(norm)));
            RKernel.at(i) /= std::sqrt(std::abs(norm));
        }

        auto [Combiner, combiner_ind] = itensor::combiner(common_inds);
        for (int i = 0; i < ker; ++i) {
            LKernel.at(i) *= Combiner;
            RKernel.at(i) *= Combiner;
        }

        ProjL *= Combiner;
        ProjR *= Combiner;

        ProjL.permute({combiner_ind, proj_idx});
        ProjR.permute({combiner_ind, proj_idx});
        for (int i = 0; i < ker; ++i) {
            for (int j = 1; j <= itensor::dim(combiner_ind); ++j) {
                ProjR.set(j, rank-i, itensor::eltC(RKernel.at(i), j));
                ProjL.set(j, rank-i, itensor::eltC(LKernel.at(i), j));
            }
        }

        ProjL *= Combiner;
        ProjR *= Combiner;
    }

    /**
     @brief Obtain a projector between ITensors A and B based on singular value decomposition
     @param [in] A : Input ITensor A
     @param [in] B : Input ITensor B
     @param [in] proj_tags : Tags to be set for an index that specifies a mode in a projector
     @param [in] maxdim : Optional. Maximum dimension used for singular value decomposition. The value 0 corresponds to the allowable maximum rank. The default value is 0.
     @param [in] pseudo_inverse_threshold : Optional. Threshold value used for computing the pseudo inverse for obtaining a projector. The singular value less than (the largest singular value) *  \a pseudo_inverse_threshold is truncated. The default value is 1e-14.
     @return Tuple composed of {LeftProjector, RightProjector, obtained singular values, number of truncated modes}
     @details For two ITensors A and B that can be producted, this function generates projectors L and R that satisfy
     \f[
     A B \simeq A L R B.
     \f]
     If the truncation does not occur during the computation of the pseudo inverse, the equality holds. The input tag \a proj_tags is set to the index connecting the projectors L and R. 
     */
    inline std::tuple<itensor::ITensor, itensor::ITensor, std::vector<double>, int> projector(const itensor::ITensor &A, const itensor::ITensor &B, const itensor::TagSet &proj_tags, int maxdim = 0, double pseudo_inverse_threshold=1e-14) {

        auto U_inds = itensor::uniqueInds(A, B);
        auto V_inds = itensor::uniqueInds(B, A);
        auto common_inds = itensor::commonInds(A, B);
        int m = 1, k = 1, n = 1;
        for (auto &x : U_inds) {
            m *= itensor::dim(x);
        }
        for (auto &x : common_inds) {
            k *= itensor::dim(x);
        }
        for (auto &x : V_inds) {
            n *= itensor::dim(x);
        }
        int rank = std::min({m, k, n});
        itensor::ITensor AB = A*B;

        itensor::Args args("LeftTags", proj_tags, "SVDMethod", "gesdd");
        if (maxdim == 0) {
            args += {"MaxDim", rank, "MinDim", rank};
        } else {
            args += {"MaxDim", std::min(maxdim, rank), "MinDim", std::min(maxdim, rank)};
        }
        auto [U, S, V] = itensor::svd(AB, U_inds, args);
        auto us_idx = itensor::commonIndex(U, S);
        auto sv_idx = itensor::commonIndex(S, V);
        int mid_dim = k;
        auto proj_idx = itensor::Index(mid_dim, proj_tags);
        itensor::ITensor InvS_U(us_idx, proj_idx), InvS_V(sv_idx, proj_idx);
        double pinv_threshold = itensor::elt(S, 1, 1) * pseudo_inverse_threshold;
        int n_svals = us_idx.dim();
        int ker = 0;
        std::vector<double> svals;
        svals.reserve(mid_dim);
        for (int i=1; i <= mid_dim; ++i) {
            double singular_val = 0.0;
            if (i <= n_svals) {
                singular_val = itensor::elt(S, i, i);
            }
            if (singular_val > pinv_threshold) {
                svals.push_back(singular_val);
                InvS_U.set(i, i, 1.0 / singular_val);
                InvS_V.set(i, i, 1.0);
            } else {
                ++ker;
            }
        }
        auto ProjR = A * U.dag() * InvS_U;
        auto ProjL = B * V.dag() * InvS_V;

        if (ker > 0) {
            fill_kernel(ProjL, ProjR, ker, proj_tags);
        }
        reorthogonalize(ProjL, ProjR, proj_tags);

        return {ProjL, ProjR, svals, ker};
    }
}
