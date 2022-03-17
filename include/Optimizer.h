/**
 * @file Optimizer.cc
 * @author wgq
 * @brief
 * @version 0.1
 * @date 2022-03-17
 * 学习g2o的优化库的使用
 * @copyright Copyright (c) 2022
 *
 */

#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/sparse_block_matrix.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_gauss_newton.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"

namespace BASIC
{
    class Optimizer
    {
    public:
        void static BundleAdjustment();
    };
}

#endif // OPTIMIZER_H