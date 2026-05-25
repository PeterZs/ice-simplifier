#include <igl/read_triangle_mesh.h>
#include <igl/write_triangle_mesh.h>
#include <igl/slice.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <build_intrinsic_info.h>
#include <coarsen_mesh.h>
#include <remove_unreferenced_intrinsic.h>
#include <connected_components.h>

#include "write_cfmap.h"
#include "write_geodesics.h"
#include "remove_degenerate_faces.h"
#include "remove_isolated_components.h"

#include <iostream>
#include <algorithm>
#include <map>
#include <vector>
#include <string>

int main(int argc, char* argv[]) {
    using namespace Eigen;

    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <input.obj> <output.obj> [n_coarse_vertices=500]" << std::endl;
        return 1;
    }

    std::string input_path  = argv[1];
    std::string output_path = argv[2];
    int n_coarse_vertices   = (argc >= 4) ? std::stoi(argv[3]) : 500;

    // Load mesh
    MatrixXd VO;
    MatrixXi FO;
    if (!igl::read_triangle_mesh(input_path, VO, FO)) {
        std::cerr << "Error: failed to read " << input_path << std::endl;
        return 1;
    }
    std::cout << "Loaded: " << VO.rows() << " vertices, " << FO.rows() << " faces" << std::endl;

    if (n_coarse_vertices <= 0) {
        std::cerr << "Error: n_coarse_vertices must be positive" << std::endl;
        return 1;
    }
    if (n_coarse_vertices >= VO.rows()) {
        std::cout << "Warning: n_coarse_vertices (" << n_coarse_vertices
                  << ") >= input vertex count (" << VO.rows() << "). Nothing to do." << std::endl;
        igl::write_triangle_mesh(output_path, VO, FO);
        return 0;
    }

    // Build intrinsic representation
    MatrixXi F = FO;
    MatrixXi G;
    MatrixXd l, A;
    MatrixXi v2fs;
    build_intrinsic_info(VO, FO, G, l, A, v2fs);

    // Check connectivity
    VectorXi v_ids, f_ids;
    int n_components;
    connected_components(FO, G, n_components, v_ids, f_ids);
    if (n_components != 1) {
        std::cout << "Warning: mesh has " << n_components << " connected components." << std::endl;
    }

    // Coarsen
    int total_removal  = VO.rows() - n_coarse_vertices;
    double weight      = 0.0; // 0=curvature error, 1=area error
    MatrixXd BC;
    std::vector<std::vector<int>> F2V;

    std::cout << "Coarsening: removing " << total_removal << " vertices..." << std::endl;
    coarsen_mesh(total_removal, weight, F, G, l, A, v2fs, BC, F2V);

    // Extract coarse mesh vertices/faces (full overload: also reindexes F2V)
    std::map<int,int> IMV, IMF;
    VectorXi vIdx, fIdx;
    remove_unreferenced_intrinsic(F, G, l, A, v2fs, F2V, IMV, IMF, vIdx, fIdx);

    MatrixXd V_coarse;
    igl::slice(VO, vIdx, 1, V_coarse);
    // F is already re-indexed by remove_unreferenced_intrinsic
    MatrixXi F_coarse = F;

    remove_degenerate_faces(F_coarse, F2V, IMV, BC);
    remove_isolated_components(F_coarse, V_coarse, VO, F2V, IMV);

    // Compact vertex array: remove vertices no longer referenced by any face
    // (orphaned by degenerate-face removal or isolated-component removal above).
    {
        int nV = (int)V_coarse.rows();
        std::vector<bool> ref(nV, false);
        for (int f = 0; f < F_coarse.rows(); f++)
            for (int k = 0; k < 3; k++) ref[F_coarse(f,k)] = true;
        int nRef = (int)std::count(ref.begin(), ref.end(), true);
        if (nRef < nV) {
            std::vector<int> o2n(nV, -1), n2o;
            n2o.reserve(nRef);
            for (int i = 0; i < nV; i++) if (ref[i]) { o2n[i] = (int)n2o.size(); n2o.push_back(i); }
            // Update F_coarse indices
            for (int f = 0; f < F_coarse.rows(); f++)
                for (int k = 0; k < 3; k++) F_coarse(f,k) = o2n[F_coarse(f,k)];
            // Compact V_coarse
            Eigen::MatrixXd Vc(nRef, 3);
            for (int i = 0; i < nRef; i++) Vc.row(i) = V_coarse.row(n2o[i]);
            V_coarse = Vc;
            // Update IMV: remap or fall back to nearest if pointing to removed vertex
            for (auto& kv : IMV) {
                int cv = kv.second;
                if (cv >= 0 && cv < nV && o2n[cv] >= 0) {
                    kv.second = o2n[cv];
                } else {
                    // IMV pointed to an orphaned vertex — find nearest remaining
                    int v = kv.first;
                    int best = 0; double bestD = 1e30;
                    for (int c = 0; c < nRef; c++) {
                        double d = (VO.row(v) - V_coarse.row(c)).squaredNorm();
                        if (d < bestD) { bestD = d; best = c; }
                    }
                    kv.second = best;
                }
            }
            std::cerr << "Compacted: removed " << (nV - nRef) << " unreferenced vertex/vertices\n";
        }
    }

    std::cout << "Coarsened: " << V_coarse.rows() << " vertices, " << F_coarse.rows() << " faces" << std::endl;

    // Write output
    if (!igl::write_triangle_mesh(output_path, V_coarse, F_coarse)) {
        std::cerr << "Error: failed to write " << output_path << std::endl;
        return 1;
    }
    std::cout << "Written: " << output_path << std::endl;

    if (!write_cfmap(output_path, (int)VO.rows(), V_coarse, F_coarse, BC, F2V, IMV))
        return 1;

    if (!write_geodesics(output_path, VO, FO, vIdx))
        return 1;

    return 0;
}
