/**
@file tensor_storage.hpp
@brief Header file for the class that manages the storage of tensors used in tensor network Monte Carlo
*/
#pragma once
#include <itensor/all_mps.h>
#include "tnmc_projector.hpp"

/**
@namespace tensor_storage
@brief Namespace containg the class that manages the storages of tensors
*/
namespace tensor_storage {
    /**
    @brief Class that manages the storages of tensors
    @details This class manages four storages of tensors used in tensor network Monte Carlo, i.e., the storages for local MPSs, local gates, truncated projectors, and full-rank projectors. The following figure summarizes the roles of these tensors.
    The truncated projectors at site n and n+1 are obtained from the full-rank projectors at site n. Therefore, for N-site systems, site n index runs from 0 to N-1 for local MPSs, local gates, and truncated projectors in 0-based indexing while n runs from 0 to N-2 for the full-rank projectors.
    */
    class tensor_storage {
        private:
            std::vector<itensor::ITensor> psi_storage_, gate_storage_, truncated_storage_;
            std::vector<tnmc_projector::projector> projector_storage_;
            std::vector<std::vector<int>> selected_modes_;
            int N_, Nt_;
        public:
            /**
            @brief Defalult constructor for an empty storage
            @details Constructor for an empty storage
            */
            tensor_storage() {}
            /**
            @brief Constructor for the storage of tensors used for tensor network Monte Carlo
            @details Constructor for the storage of tensors used for tensor network Monte Carlo. The parameter \a psi specifies an initial state and \a Nt corresponds to the depth of a circuit.
            @param [in] psi : itensor::MPS representing an initial state
            @param [in] Nt : Depth of a circuit simulated in tensor network Monte Carlo
            */
            tensor_storage(itensor::MPS &psi, int Nt);

            /**
            @brief Index calculator for a local MPS at site \a n and depth \a t
            @details Index calculator for a local MPS at site \a n and depth \a t. This index can be also used for the storages of local gates and truncated projectors.
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Index for a target tensor in the storage
            */
            int calc_psi_index(int n, int t);
            /**
            @brief Index calculator for a projector at site \a n and depth \a t
            @details Index calculator for a projector between site \a n and \a n+1 at depth \a t.
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Index for a target projector in the storage
            */
            int calc_projector_index(int n, int t);

            /**
            @brief Reference for a truncated projector at site \a n and depth \a t
            @details Reference for a truncated projector between site \a n and \a n+1 at depth \a t.
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Reference for a target truncated projector in the storage
            */
            itensor::ITensor& get_truncated_ref(int n, int t) {
                return truncated_storage_.at(calc_psi_index(n, t));
            };

            /**
            @brief Reference for a truncated projector at index \a n
            @details Reference for a truncated projector at index \a n
            @param [in] n : Index for the storage (0-indexed)
            @return Reference for a target truncated projector in the storage
            */
            itensor::ITensor& get_truncated_ref(int n) {
                return truncated_storage_.at(n);
            };

            /**
            @brief Truncated projector at site \a n and depth \a t
            @details Truncated projector at site \a n and depth \a t. This function returns the copy of the truncated projector.
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Tensor corresponding to the target truncated projector in the storage
            */
            itensor::ITensor get_truncated(int n, int t) {
                return truncated_storage_.at(calc_psi_index(n, t));
            };

            /**
            @brief Truncated projector at index \a n
            @details Truncated projector at index \a n. This function returns the copy of the truncated projector.
            @param [in] n : Index for the storage (0-indexed)
            @return Tensor corresponding to the target truncated projector in the storage
            */
            itensor::ITensor get_truncated(int n) {
                return truncated_storage_.at(n);
            };

            /**
            @brief Set truncated projector at site \a n and depth \a t
            @details Set truncated projector at site \a n and depth \a t
            @param [in] truncated_proj : Input truncated projector
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            */
            void set_truncated(const itensor::ITensor &truncated_proj, int n, int t) {
                truncated_storage_.at(calc_psi_index(n, t)) = truncated_proj;
            };

            /**
            @brief Set truncated projector at index \a n
            @details Set truncated projector at index \a n
            @param [in] truncated_proj : Input truncated projector
            @param [in] n : Index for the storage (0-indexed)
            */
            void set_truncated(const itensor::ITensor &truncated_proj, int n) {
                truncated_storage_.at(n) = truncated_proj;
            };

            /**
            @brief Reference for a local MPS at site \a n and depth \a t
            @details Reference for a local MPS at site \a n and depth \a t
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Reference for a local MPS in the storage
            */
            itensor::ITensor& get_psi_ref(int n, int t) {
                return psi_storage_.at(calc_psi_index(n, t));
            };

            /**
            @brief Reference for a local MPS at index \a n
            @details Reference for a local MPS at index \a n
            @param [in] n : Index for the storage (0-indexed)
            @return Reference for a local MPS in the storage
            */
            itensor::ITensor& get_psi_ref(int n) {
                return psi_storage_.at(n);
            };

            /**
            @brief Set a local MPS at site \a n and depth \a t
            @details Set a local MPS at site \a n and depth \a t
            @param [in] psi : Input local MPS for site \a n and depth \a t 
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            */
            void set_psi(const itensor::ITensor &psi, int n, int t) {
                psi_storage_.at(calc_psi_index(n, t)) = psi;
            };

            /**
            @brief Set a local MPS at index \a n
            @details Set a local MPS at index \a n
            @param [in] psi : Input local MPS for site \a n and depth \a t 
            @param [in] n : Index for the storage (0-indexed)
            */
            void set_psi(const itensor::ITensor &psi, int n) {
                psi_storage_.at(n) = psi;
            };

            /**
            @brief Local MPS at site \a n and depth \a t
            @details Local MPS at site \a n and depth \a t. This function returns the copy of the local MPS.
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Local MPS at site \a n and depth \a t
            */
            itensor::ITensor get_psi(int n, int t) {
                return get_psi_ref(n, t);
            };

            /**
            @brief Local MPS at index \a n
            @details Local MPS at index \a n. This function returns the copy of the local MPS.
            @param [in] n : Index for the storage (0-indexed)
            @return Local MPS at index \a n
            */
            itensor::ITensor get_psi(int n) {
                return get_psi_ref(n);
            };

            /**
            @brief Reference for a gate at site \a n and depth \a t
            @details Reference for a gate at site \a n and depth \a t
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Reference for the gate at site \a n and depth \a t
            */
            itensor::ITensor& get_gate_ref(int n, int t) {
                return gate_storage_.at(calc_psi_index(n, t));
            };

            /**
            @brief Reference for a gate at index \a n
            @details Reference for a gate at index \a n
            @param [in] n : Index for the storage (0-indexed)
            @return Reference for the gate at index \a n
            */
            itensor::ITensor& get_gate_ref(int n) {
                return gate_storage_.at(n);
            };

            /**
            @brief Set a gate at site \a n and depth \a t
            @details Set a gate at site \a n and depth \a t
            @param [in] gate : Tensor corresponding to a local gate
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            */
            void set_gate(itensor::ITensor &gate, int n, int t) {
                gate_storage_.at(calc_psi_index(n, t)) = gate;
            };

            /**
            @brief Set a gate at index \a n
            @details Set a gate at index \a n
            @param [in] gate : Tensor corresponding to a local gate
            @param [in] n : Index for the storage (0-indexed)
            */
            void set_gate(itensor::ITensor &gate, int n) {
                gate_storage_.at(n) = gate;
            };

            /**
            @brief Gate at site \a n and depth \a t
            @details Gate at site \a n and depth \a t. This function returns the copy of the gate
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Tensor corresponding to the gate at site \a n and depth \a t
            */
            itensor::ITensor get_gate(int n, int t) {
                return get_gate_ref(n, t);
            };

            /**
            @brief Gate at index \a n
            @details Gate at index \a n. This function returns the copy of the gate
            @param [in] n : Index for the storage (0-indexed)
            @return Tensor corresponding to the gate at index \a n
            */
            itensor::ITensor get_gate(int n) {
                return get_gate_ref(n);
            };

            /**
            @brief Set a projector between sites \a n and \a n+1 at depth \a t
            @details Set a projector between sites \a n and \a n+1 at depth \a t
            @param [in] proj : Input tnmc_projector::projector
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            */
            void set_projector(const tnmc_projector::projector &proj, int n, int t) {
                projector_storage_.at(calc_projector_index(n, t)) = proj;
            }

            /**
            @brief Set a projector at index \a n
            @details Set a projector at index \a n
            @param [in] proj : Input tnmc_projector::projector
            @param [in] n : Index for the storage (0-indexed)
            */
            void set_projector(const tnmc_projector::projector &proj, int n) {
                projector_storage_.at(n) = proj;
            }

            /**
            @brief Update a truncated projector between sites \a n and \a n+1 at depth \a t by choosing the mode specified by an index \a select
            @details Update a truncated projector between sites \a n and \a n+1 at depth \a t by choosing the mode specified by an index \a select
            @param [in] select : Index specifying a newly selected mode
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            */
            void reselect_modes(const std::vector<int> &select, int n, int t);

            /**
            @brief Index for the present selected mode for the truncated projector between sites \a n and \a n+1 at depth \a t
            @details Index for the present selected mode for the truncated projector between sites \a n and \a n+1 at depth \a t
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Index for the present selected mode
            */
            std::vector<int> get_selected_modes(int n, int t) {
                return selected_modes_.at(calc_projector_index(n, t));
            }

            /**
            @brief Index for the present selected mode for the truncated projector at index \a n
            @details Index for the present selected mode for the truncated projector at index \a n
            @param [in] n : Index for the storage (0-indexed)
            @return Index for the present selected mode
            */
            std::vector<int> get_selected_modes(int n) {
                return selected_modes_.at(n);
            }

            void set_selected_modes(const std::vector<int> &selected, int n) {
                selected_modes_.at(n) = selected;
            }

            /**
            @brief Number of modes for the projector between sites \a n and \a n+1 at depth \a t
            @details Number of modes for the projector between sites \a n and \a n+1 at depth \a t
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Number of modes for the projector, i.e., the rank of the full projector
            */
            int projector_size(int n, int t) {
                return projector_storage_.at(calc_projector_index(n, t)).mode_n();
            }

            /**
            @brief Number of modes for the projector at index \a n
            @details Number of modes for the projector at index \a n
            @param [in] n : Index for the storage (0-indexed)
            @return Number of modes for the projector, i.e., the rank of the full projector
            */
            int projector_size(int n) {
                return projector_storage_.at(n).mode_n();
            }

            /**
            @brief Rank of the truncated projector between sites \a n and \a n+1 at depth \a t
            @details Rank of the truncated projector between sites \a n and \a n+1 at depth \a t
            @param [in] n : Site number (0-indexed)
            @param [in] t : Depth number (0-indexed)
            @return Rank of the truncated projector
            */
            int projector_rank(int n, int t) {
                return projector_storage_.at(calc_projector_index(n, t)).rank();
            }

            /**
            @brief Rank of the truncated projector at index \a n
            @details Rank of the truncated projector at index \a n
            @param [in] n : Index for the storage (0-indexed)
            @return Rank of the truncated projector
            */
            int projector_rank(int n) {
                return projector_storage_.at(n).rank();
            }

            /**
            @brief Size of the storage for local MPSs
            @details Size of the storage for local MPSs
            @return Size (the number of elements) of the storage for local MPSs
            */
            int psi_storage_size() {
                return psi_storage_.size();
            }

            /**
            @brief Size of the storage for local gates
            @details Size of the storage for local gates
            @return Size (the number of elements) of the storage for local gates
            */
            int gate_storage_size() {
                return gate_storage_.size();
            }

            /**
            @brief Size of the storage for projectors
            @details Size of the storage for projectors
            @return Size (the number of elements) of the storage for projectors
            */
            int projector_storage_size() {
                return projector_storage_.size();
            }
    };


    inline tensor_storage::tensor_storage(itensor::MPS &psi, int Nt) : N_(itensor::length(psi)), Nt_(Nt) {
        int psi_dim = N_ * (Nt_+1);
        psi_storage_ = std::vector<itensor::ITensor>(psi_dim);
        for (int i = 0; i < N_; ++i) {
            set_psi(psi.Aref(i+1), i, 0);
        }

        int gate_dim = N_ * Nt_;
        gate_storage_ = std::vector<itensor::ITensor>(gate_dim);

        int proj_dim = (N_-1) * Nt_;
        projector_storage_ = std::vector<tnmc_projector::projector>(proj_dim);
        truncated_storage_ = std::vector<itensor::ITensor>(gate_dim);
        selected_modes_ = std::vector<std::vector<int>>(proj_dim);
    }

    inline int tensor_storage::calc_psi_index(int n, int t) {
        return N_ * t + n;
    }

    inline int tensor_storage::calc_projector_index(int n, int t) {
        return (N_ - 1)*t + n;
    }

    inline void tensor_storage::reselect_modes(const std::vector<int> &select, int n, int t) {
        int projector_idx = calc_projector_index(n, t);
        auto& projector = projector_storage_.at(projector_idx);
        int rank = projector.rank();
        itensor::ITensor left_truncated = projector.left_projector_;
        itensor::ITensor right_truncated = projector.right_projector_;
        itensor::Index proj_idx = itensor::findIndex(left_truncated, "Proj");
        itensor::Index select_idx = itensor::Index(rank, "Proj");
        itensor::ITensor selector(proj_idx, select_idx);
        for (int i = 1; i <= rank; ++i) {
            selector.set(select.at(i-1), i, 1.0);
        }
        left_truncated *= selector;
        right_truncated *= selector;

        truncated_storage_.at(calc_psi_index(n, t)) = left_truncated;
        truncated_storage_.at(calc_psi_index(n+1, t)) = right_truncated;
        selected_modes_.at(projector_idx) = select;
    }
}
