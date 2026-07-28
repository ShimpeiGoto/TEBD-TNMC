/**
 @file tnmc_projector.hpp
 @brief Header file for the class manages a projector
*/
#pragma once
#include "itensor/itensor.h"
#include <itensor/all_basic.h>

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
            int rank_, mode_n_;
            friend class tensor_storage::tensor_storage;
        public:
            projector(const itensor::ITensor &Q, int rank_in);
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
    };

    /**
      @brief Constructor for a projector
      @param [in] left : Left side projector
      @param [in] right : Right side projector
      @param [in] rank_in : Rank of a truncated projector
      @param [in] svals : (Modified) singular values obtained during singular value decomposition
      @details For the truncation of projector, the modes corresponding to \a rank - 1 largest singular values are always chosen. The last one mode is chosen with a probability proportional to its (modified) singular value.

      */
    inline projector::projector(const itensor::ITensor &Q, int rank_in) : left_projector_(Q), right_projector_(itensor::dag(Q)), rank_(rank_in), mode_n_(itensor::dim(itensor::findIndex(Q, "Proj"))) {
    }

    /**
      @brief Default constructor for an empty projector
      @details This constructor requires for initializing STL continers.
      */
    inline projector::projector() {
        left_projector_ = itensor::ITensor();
        right_projector_ = itensor::ITensor();
    }
}
