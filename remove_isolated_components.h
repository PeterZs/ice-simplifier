#pragma once

#include <Eigen/Core>

#include <algorithm>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

// Remove isolated small connected components from the coarse mesh.
//
// Components with fewer than kMinCompFaces faces are dropped. Fine vertices
// stranded in a removed component are reassigned to the nearest main-component
// coarse vertex via IMV.
//
// Returns the number of faces removed (0 = single connected component).
inline int remove_isolated_components(
    Eigen::MatrixXi&               F_coarse,
    const Eigen::MatrixXd&         V_coarse,
    const Eigen::MatrixXd&         VO,
    std::vector<std::vector<int>>& F2V,
    std::map<int,int>&             IMV,
    int                            kMinCompFaces = 4)
{
    int nF = (int)F_coarse.rows();

    // Build edge-to-face adjacency
    std::map<std::pair<int,int>, std::vector<int>> edgeFaces;
    for (int f = 0; f < nF; f++)
        for (int k = 0; k < 3; k++) {
            int a = F_coarse(f,k), b = F_coarse(f,(k+1)%3);
            if (a > b) std::swap(a, b);
            edgeFaces[{a,b}].push_back(f);
        }

    // BFS: label connected components
    std::vector<int> comp(nF, -1);
    int nComp = 0;
    std::vector<int> compSz;
    for (int s = 0; s < nF; s++) {
        if (comp[s] >= 0) continue;
        comp[s] = nComp;
        std::vector<int> stk = {s};
        int sz = 1;
        while (!stk.empty()) {
            int f = stk.back(); stk.pop_back();
            for (int k = 0; k < 3; k++) {
                int a = F_coarse(f,k), b = F_coarse(f,(k+1)%3);
                if (a > b) std::swap(a, b);
                for (int nb : edgeFaces[{a,b}])
                    if (comp[nb] < 0) { comp[nb] = nComp; sz++; stk.push_back(nb); }
            }
        }
        compSz.push_back(sz);
        nComp++;
    }

    int mainComp = (int)(std::max_element(compSz.begin(), compSz.end()) - compSz.begin());

    // Collect main-component vertex set for nearest-vertex reassignment
    int nV = (int)V_coarse.rows();
    std::vector<bool> isMainVert(nV, false);
    for (int f = 0; f < nF; f++)
        if (comp[f] == mainComp)
            for (int k = 0; k < 3; k++) isMainVert[F_coarse(f,k)] = true;

    // Reassign fine vertices from small components to nearest main-component coarse vertex
    int nRemovedFaces = 0;
    for (int f = 0; f < nF; f++) {
        if (comp[f] == mainComp || compSz[comp[f]] >= kMinCompFaces) continue;
        for (int v : F2V[f]) {
            int bestCV = 0; double bestD = 1e30;
            for (int cv = 0; cv < nV; cv++) {
                if (!isMainVert[cv]) continue;
                double d = (VO.row(v) - V_coarse.row(cv)).squaredNorm();
                if (d < bestD) { bestD = d; bestCV = cv; }
            }
            IMV[v] = bestCV;
        }
        F2V[f].clear();
        nRemovedFaces++;
    }

    if (nRemovedFaces == 0) return 0;

    std::cerr << "Warning: removed " << nRemovedFaces
              << " face(s) from isolated small component(s)\n";

    std::vector<int> keep;
    keep.reserve(nF - nRemovedFaces);
    for (int f = 0; f < nF; f++)
        if (comp[f] == mainComp || compSz[comp[f]] >= kMinCompFaces) keep.push_back(f);

    Eigen::MatrixXi Fc((int)keep.size(), 3);
    std::vector<std::vector<int>> F2Vc;
    F2Vc.reserve(keep.size());
    for (int i = 0; i < (int)keep.size(); i++) {
        Fc.row(i) = F_coarse.row(keep[i]);
        F2Vc.push_back(std::move(F2V[keep[i]]));
    }
    F_coarse = Fc;
    F2V      = std::move(F2Vc);

    return nRemovedFaces;
}
