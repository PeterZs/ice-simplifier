#pragma once

#include <Eigen/Core>

#include <iostream>
#include <map>
#include <vector>

// Remove degenerate faces (duplicate vertex indices) from the coarse mesh.
//
// Degenerate faces arise from intrinsic topology degradation during heavy
// decimation (paper Sec. 3.1 "self-faces", Sec. 6.4 "near-degenerate
// triangles constructed during simplification").  They produce zero-length
// edges whose XPBD spring constraints are singular.
//
// Fine vertices stranded in a degenerate face are reassigned to their
// nearest coarse vertex via argmax(BC), inserted into IMV so that
// write_cfmap emits a valid nearestCoarse entry for every fine vertex.
//
// Returns the number of degenerate faces removed (0 = clean mesh).
inline int remove_degenerate_faces(
    Eigen::MatrixXi&               F_coarse,
    std::vector<std::vector<int>>& F2V,
    std::map<int,int>&             IMV,
    const Eigen::MatrixXd&         BC)
{
    int n_degen = 0;
    for (int f = 0; f < F_coarse.rows(); f++) {
        int v0 = F_coarse(f,0), v1 = F_coarse(f,1), v2 = F_coarse(f,2);
        if (v0 == v1 || v1 == v2 || v2 == v0) {
            for (int v : F2V[f]) {
                int k = 0;
                if (BC(v,1) > BC(v,k)) k = 1;
                if (BC(v,2) > BC(v,k)) k = 2;
                IMV[v] = F_coarse(f, k);
            }
            F2V[f].clear();
            n_degen++;
        }
    }

    if (n_degen == 0) return 0;

    std::cerr << "Warning: removed " << n_degen << " degenerate face(s) from output\n";

    std::vector<int> keep;
    keep.reserve(F_coarse.rows() - n_degen);
    for (int f = 0; f < F_coarse.rows(); f++) {
        int v0 = F_coarse(f,0), v1 = F_coarse(f,1), v2 = F_coarse(f,2);
        if (v0 != v1 && v1 != v2 && v2 != v0) keep.push_back(f);
    }

    Eigen::MatrixXi Fc(keep.size(), 3);
    std::vector<std::vector<int>> F2Vc;
    F2Vc.reserve(keep.size());
    for (int i = 0; i < (int)keep.size(); i++) {
        Fc.row(i) = F_coarse.row(keep[i]);
        F2Vc.push_back(std::move(F2V[keep[i]]));
    }
    F_coarse = Fc;
    F2V      = std::move(F2Vc);

    return n_degen;
}
