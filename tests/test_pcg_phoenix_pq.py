"""
Unit tests for Toyota Research Institute `diffmpc` zero-float p/q Schur-Complement
PCG solver and integrator sub-stepping / Midpoint / RK4 fixes.
"""

import importlib.util
import os
import sys
import unittest

_MOD_PATH = os.path.join(
    os.path.dirname(__file__), "..", "diffmpc", "utils", "pcg_phoenix_pq.py"
)
_SPEC = importlib.util.spec_from_file_location("pcg_phoenix_pq", _MOD_PATH)
pcg_mod = importlib.util.module_from_spec(_SPEC)
sys.modules["pcg_phoenix_pq"] = pcg_mod
assert _SPEC.loader is not None
_SPEC.loader.exec_module(pcg_mod)

SchurPcgSolverPq = pcg_mod.SchurPcgSolverPq


class TestToyotaDiffMpcPq(unittest.TestCase):
    def test_schur_pcg_spd_tridiagonal_convergence(self):
        # 8x8 block-tridiagonal SPD Schur complement matrix S and Jacobi preconditioner Phi^{-1}
        n = 8
        s_mat = [[0.0] * n for _ in range(n)]
        phi_inv = [[0.0] * n for _ in range(n)]
        for i in range(n):
            s_mat[i][i] = 4.0
            phi_inv[i][i] = 0.25
            if i + 1 < n:
                s_mat[i][i + 1] = -1.0
                s_mat[i + 1][i] = -1.0
        gamma = [1.0, -2.0, 3.0, -1.5, 2.5, -0.5, 1.25, -1.0]

        res = SchurPcgSolverPq.solve_spd_pq(
            s_mat, gamma, phi_inv, tol_num=1, tol_den=10000, max_iter=32
        )
        self.assertTrue(res.converged)
        self.assertLessEqual(res.num_iterations, n)
        p_err, q_err = res.residual_inf_pq
        self.assertLess(p_err / q_err, 1e-4)

        # Verify S * lam ~= gamma
        lam = [v / float(SchurPcgSolverPq.Q20_ONE) for v in res.kkt_multipliers_q20]
        for i in range(n):
            lhs = sum(s_mat[i][j] * lam[j] for j in range(n))
            self.assertAlmostEqual(lhs, gamma[i], places=4)


if __name__ == "__main__":
    unittest.main()
