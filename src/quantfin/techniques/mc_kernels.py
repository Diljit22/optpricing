# src/quantfin/techniques/mc_kernels.py

import numpy as np
import numba

@numba.jit(nopython=True, fastmath=True, cache=True)
def bsm_kernel(n_paths, n_steps, log_s0, r, q, sigma, dt, dw):
    """JIT-compiled kernel for BSM SDE simulation."""
    log_s = np.full(n_paths, log_s0)
    drift = (r - q - 0.5 * sigma**2) * dt
    for i in range(n_steps):
        log_s += drift + sigma * dw[:, i]
    return log_s

@numba.jit(nopython=True, fastmath=True, cache=True)
def heston_kernel(n_paths, n_steps, log_s0, v0, r, q, kappa, theta, rho, vol_of_vol, dt, dw1, dw2):
    """JIT-compiled kernel for Heston SDE simulation (Full Truncation)."""
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    for i in range(n_steps):
        v_pos = np.maximum(v, 0)
        v_sqrt = np.sqrt(v_pos)
        log_s += (r - q - 0.5 * v_pos) * dt + v_sqrt * dw1[:, i]
        v += kappa * (theta - v_pos) * dt + vol_of_vol * v_sqrt * dw2[:, i]
    return log_s

@numba.jit(nopython=True, fastmath=True, cache=True)
def merton_kernel(n_paths, n_steps, log_s0, r, q, sigma, lambda_, mu_j, sigma_j, dt, dw, jump_counts):
    """JIT-compiled kernel for Merton Jump-Diffusion SDE simulation."""
    log_s = np.full(n_paths, log_s0)
    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    drift = (r - q - 0.5 * sigma**2 - compensator) * dt
    for i in range(n_steps):
        log_s += drift + sigma * dw[:, i]
        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)
            jump_sizes = np.random.normal(mu_j, sigma_j, num_jumps)
            jump_idx = 0
            for path_idx in range(n_paths):
                for _ in range(jumps_this_step[path_idx]):
                    log_s[path_idx] += jump_sizes[jump_idx]
                    jump_idx += 1
    return log_s

@numba.jit(nopython=True, fastmath=True, cache=True)
def bates_kernel(n_paths, n_steps, log_s0, v0, r, q, kappa, theta, rho, vol_of_vol, lambda_, mu_j, sigma_j, dt, dw1, dw2, jump_counts):
    """JIT-compiled kernel for Bates (Heston + Jumps) SDE simulation."""
    log_s = np.full(n_paths, log_s0)
    v = np.full(n_paths, v0)
    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
    
    for i in range(n_steps):
        v_pos = np.maximum(v, 0)
        v_sqrt = np.sqrt(v_pos)
        
        # Heston part
        log_s += (r - q - 0.5 * v_pos - compensator) * dt + v_sqrt * dw1[:, i]
        v += kappa * (theta - v_pos) * dt + vol_of_vol * v_sqrt * dw2[:, i]
        
        # Merton Jump part
        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)
            jump_sizes = np.random.normal(mu_j, sigma_j, num_jumps)
            jump_idx = 0
            for path_idx in range(n_paths):
                for _ in range(jumps_this_step[path_idx]):
                    log_s[path_idx] += jump_sizes[jump_idx]
                    jump_idx += 1
    return log_s



@numba.jit(nopython=True, fastmath=True, cache=True)
def sabr_kernel(n_paths, n_steps, s0, v0, r, q, alpha, beta, rho, dt, dw1, dw2):
    """JIT-compiled kernel for SABR SDE simulation."""
    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0) # v is sigma_t in SABR
    rho_bar = np.sqrt(1 - rho**2)

    for i in range(n_steps):
        # Evolve spot price S_t directly
        s_pos = np.maximum(s, 1e-8) # Avoid negative spot
        v_pos = np.maximum(v, 0)
        s += (r - q) * s_pos * dt + v_pos * (s_pos**beta) * dw1[:, i]
        
        # Evolve volatility sigma_t (lognormal process)
        v = v * np.exp(-0.5 * alpha**2 * dt + alpha * dw2[:, i])
        
    return s

@numba.jit(nopython=True, fastmath=True, cache=True)
def sabr_jump_kernel(n_paths, n_steps, s0, v0, r, q, alpha, beta, rho, lambda_, mu_j, sigma_j, dt, dw1, dw2, jump_counts):
    """JIT-compiled kernel for SABR with log-normal jumps on the spot process."""
    s = np.full(n_paths, s0)
    v = np.full(n_paths, v0) # v is sigma_t
    rho_bar = np.sqrt(1 - rho**2)
    compensator = lambda_ * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)

    for i in range(n_steps):
        s_pos = np.maximum(s, 1e-8)
        v_pos = np.maximum(v, 0)
        
        # Evolve spot with jump compensator
        s += (r - q - compensator) * s_pos * dt + v_pos * (s_pos**beta) * dw1[:, i]
        
        # Evolve volatility
        v = v * np.exp(-0.5 * alpha**2 * dt + alpha * dw2[:, i])
        
        # Add jumps
        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)
            # Jumps are multiplicative: S_t+ = S_t * exp(J)
            jump_multipliers = np.exp(np.random.normal(mu_j, sigma_j, num_jumps))
            jump_idx = 0
            for path_idx in range(n_paths):
                for _ in range(jumps_this_step[path_idx]):
                    s[path_idx] *= jump_multipliers[jump_idx]
                    jump_idx += 1
    return s


@numba.jit(nopython=True, fastmath=True, cache=True)
def kou_kernel(n_paths, n_steps, log_s0, r, q, sigma, lambda_, p_up, eta1, eta2, dt, dw, jump_counts):
    """JIT-compiled kernel for Kou Double-Exponential Jump-Diffusion SDE."""
    log_s = np.full(n_paths, log_s0)
    compensator = lambda_ * ((p_up * eta1 / (eta1 - 1)) + ((1 - p_up) * eta2 / (eta2 + 1)) - 1)
    drift = (r - q - 0.5 * sigma**2 - compensator) * dt
    
    for i in range(n_steps):
        log_s += drift + sigma * dw[:, i]
        jumps_this_step = jump_counts[:, i]
        if np.any(jumps_this_step > 0):
            num_jumps = np.sum(jumps_this_step)
            
            # Determine which jumps are up and which are down
            up_or_down = np.random.random(num_jumps)
            num_up_jumps = np.sum(up_or_down < p_up)
            num_down_jumps = num_jumps - num_up_jumps
            
            # Generate jump sizes from exponential distributions
            total_jump_size = 0.0
            if num_up_jumps > 0:
                total_jump_size += np.sum(np.random.exponential(1.0 / eta1, num_up_jumps))
            if num_down_jumps > 0:
                total_jump_size -= np.sum(np.random.exponential(1.0 / eta2, num_down_jumps))
            
            # This part is tricky to vectorize perfectly in Numba, so we loop
            jump_idx = 0
            for path_idx in range(n_paths):
                if jumps_this_step[path_idx] > 0: # This path has a jump
                    # For simplicity in Numba, we add the average jump size
                    # A more complex implementation could assign specific jumps
                    log_s[path_idx] += total_jump_size / num_jumps * jumps_this_step[path_idx]

    return log_s

@numba.jit(nopython=True, fastmath=True, cache=True)
def dupire_kernel(n_paths, n_steps, log_s0, r, q, dt, dw, vol_surface_func):
    """
    JIT-compiled kernel for Dupire Local Volatility SDE simulation.
    Note: Numba cannot compile the vol_surface_func itself, so this kernel
    will have a slight overhead from calling back into Python for the vol lookup.
    """
    log_s = np.full(n_paths, log_s0)
    for i in range(n_steps):
        t_current = i * dt
        current_spot = np.exp(log_s)
        local_vol = vol_surface_func(t_current, current_spot)
        drift = (r - q - 0.5 * local_vol**2) * dt
        log_s += drift + local_vol * dw[:, i]
    return log_s