import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
import equinox as eqx


@jax.vmap
def fdtd_2d_binary_operator_paper(state_i, state_j):
    """
    Binary operator for associative scan implementing the paper's equation (10).

    This processes:
    x_n = M^DAMP^{-1} * M^EXP * x_{n-1} + M^DAMP^{-1} * F^EXP_n

    Args:
        state_i: tuple containing (x_i, F_i) at position i
        state_j: tuple containing (x_j, F_j) at position j
    Returns:
        new element (x_out, F_out)
    """
    x_i, Bu_i = state_i
    x_j, Bu_j = state_j

    # Get parameters from context
    c = jax.lax.stop_gradient(context['c'])
    kp = jax.lax.stop_gradient(context['kp'])
    k = jax.lax.stop_gradient(context['k'])
    dt = jax.lax.stop_gradient(context['dt'])
    dx = jax.lax.stop_gradient(context['dx'])
    grid_side = jax.lax.stop_gradient(context['grid_side'])

    grid_size = grid_side * grid_side

    # Split state vector into pressure and velocity components
    p_i = x_i[:grid_size]
    o_i = x_i[grid_size:].reshape(2, grid_size)

    # Step 1: Apply M^EXP (equation 11)
    # Define gradient and divergence operators
    def grad_p(p):
        """Compute gradient of pressure field"""
        p_2d = p.reshape(grid_side, grid_side)
        grad_x = (jnp.roll(p_2d, -1, axis=1) - jnp.roll(p_2d, 1, axis=1)) / (2 * dx)
        grad_y = (jnp.roll(p_2d, -1, axis=0) - jnp.roll(p_2d, 1, axis=0)) / (2 * dx)
        return jnp.stack([grad_x.reshape(-1), grad_y.reshape(-1)], axis=0)

    def div_o(o):
        """Compute divergence of velocity field"""
        ox = o[0].reshape(grid_side, grid_side)
        oy = o[1].reshape(grid_side, grid_side)
        div_x = (jnp.roll(ox, -1, axis=1) - jnp.roll(ox, 1, axis=1)) / (2 * dx)
        div_y = (jnp.roll(oy, -1, axis=0) - jnp.roll(oy, 1, axis=0)) / (2 * dx)
        return (div_x + div_y).reshape(-1)

    # Apply M^EXP * x_i (equation 11)
    grad_p_i = grad_p(p_i)
    div_o_i = div_o(o_i)

    p_exp = p_i - c ** 2 * dt * div_o_i
    o_exp = o_i - dt * grad_p_i

    # Add input forcing F^EXP_j (equation 12)
    p_exp = p_exp + dt * Bu_j

    # Step 2: Apply M^DAMP^{-1} (equation 13)
    p_damp_inv = 1.0 / (1.0 + dt * kp)
    o_damp_inv = 1.0 / (1.0 + dt * k)

    p_new = p_exp * p_damp_inv
    o_new = o_exp * o_damp_inv.reshape(1, -1)

    # Combine into state vector
    x_new = jnp.concatenate([p_new, o_new.reshape(-1)])

    return x_new, Bu_j


def apply_fdtd_2d_scan(c, kp, k, dt, dx, Bu, grid_side):
    """
    2D FDTD propagation using associative scan following paper equations.

    Args:
        c (float32): wave speed array
        kp (float32): pressure damping
        k (float32): velocity damping
        dt (float32): time step
        dx (float32): grid spacing
        Bu (float32): input forcing
        grid_side (int): grid dimension

    Returns:
        p, ox, oy: fields over time
    """
    grid_size = grid_side * grid_side

    # Ensure stability
    max_c = 0.7 * dx / (dt * jnp.sqrt(2.0))
    c = jnp.clip(c, 0.1, max_c)
    kp = nn.softplus(kp)
    k = nn.softplus(k)

    # Make parameters available to the binary operator via a global context
    global context
    context = {
        'c': c,
        'kp': kp,
        'k': k,
        'dt': dt,
        'dx': dx,
        'grid_side': grid_side
    }

    # Initial state: zeros for all components
    init_x = jnp.zeros(grid_size * 3)  # p, ox, oy

    # Create pairs for scan
    x_init = jnp.zeros_like(init_x)
    x_Bu_pairs = (jnp.repeat(x_init[None, :], Bu.shape[0], axis=0), Bu)

    # Apply scan
    final_states, _ = jax.lax.associative_scan(fdtd_2d_binary_operator_paper, x_Bu_pairs)

    # Extract fields
    p = final_states[:, :grid_size]
    ox = final_states[:, grid_size:2 * grid_size]
    oy = final_states[:, 2 * grid_size:3 * grid_size]

    return p, ox, oy


# Alternative implementation without associative scan
def apply_fdtd_2d_direct(c, kp, k, dt, dx, Bu, grid_side):
    """
    Direct implementation of the paper's equations (10)-(14).
    """
    grid_size = grid_side * grid_side
    seq_len = Bu.shape[0]

    # Ensure CFL stability
    max_c = 0.7 * dx / (dt * jnp.sqrt(2.0))
    c = jnp.clip(c, 0.1, max_c)
    kp = nn.softplus(kp)
    k = nn.softplus(k)

    # Initialize state vectors
    p = jnp.zeros((seq_len + 1, grid_size))
    ox = jnp.zeros((seq_len + 1, grid_size))
    oy = jnp.zeros((seq_len + 1, grid_size))

    # Define gradient and divergence operators
    def gradient_p(p):
        p_2d = p.reshape(grid_side, grid_side)
        grad_x = (jnp.roll(p_2d, -1, axis=1) - jnp.roll(p_2d, 1, axis=1)) / (2 * dx)
        grad_y = (jnp.roll(p_2d, -1, axis=0) - jnp.roll(p_2d, 1, axis=0)) / (2 * dx)
        return grad_x.reshape(-1), grad_y.reshape(-1)

    def divergence_o(ox, oy):
        ox_2d = ox.reshape(grid_side, grid_side)
        oy_2d = oy.reshape(grid_side, grid_side)
        div_x = (jnp.roll(ox_2d, -1, axis=1) - jnp.roll(ox_2d, 1, axis=1)) / (2 * dx)
        div_y = (jnp.roll(oy_2d, -1, axis=0) - jnp.roll(oy_2d, 1, axis=0)) / (2 * dx)
        return (div_x + div_y).reshape(-1)

    # Apply update equations (6)-(9) for each time step
    def step_fn(carry, Bu_n):
        p_prev, ox_prev, oy_prev = carry

        # Step 1: Explicit update (equations 6-7)
        grad_x, grad_y = gradient_p(p_prev)

        ox_star = ox_prev - dt * grad_x
        oy_star = oy_prev - dt * grad_y

        div_o = divergence_o(ox_star, oy_star)
        p_star = p_prev - c ** 2 * dt * div_o + dt * Bu_n

        # Step 2: Implicit damping (equations 8-9)
        ox_n = ox_star / (1.0 + dt * k)
        oy_n = oy_star / (1.0 + dt * k)
        p_n = p_star / (1.0 + dt * kp)

        return (p_n, ox_n, oy_n), (p_n, ox_n, oy_n)

    # Initial state
    init_state = (p[0], ox[0], oy[0])

    # Apply the step function to each time step
    _, results = jax.lax.scan(step_fn, init_state, Bu)

    p_out, ox_out, oy_out = results

    return p_out, ox_out, oy_out


# def initialize_c_and_k(grid_side, dt, dx, freq_min, freq_max, damping_factor, key):
#     """
#     Initialize c and k matrices based on desired frequency response with reduced variability.
#     """
#     grid_size = grid_side * grid_side
#     keys = jax.random.split(key, 3)
#
#     # Calculate required c values for desired frequencies
#     # For spatial frequency ξ = 1.0 (normalized):
#     # θ = 2π·f·dt, and c = θ·dx/(dt·π·√2)
#
#     # Create more uniform frequency distribution (less extreme)
#     alpha = jnp.log(freq_max / freq_min)
#
#     # Generate more uniform values (less random)
#     # Use linear interpolation instead of pure random for more uniformity
#     indices = jnp.arange(grid_size) / (grid_size - 1)  # Values from 0 to 1
#     # Add small random perturbation
#     random_factor = 0.3  # Reduced from typical 1.0
#     u = indices + random_factor * (jax.random.uniform(keys[0], (grid_size,)) - 0.5)
#     u = jnp.clip(u, 0.0, 1.0)  # Keep within [0, 1]
#
#     frequencies = freq_min * jnp.exp(alpha * u)
#
#     # Convert frequencies to required c values
#     theta = 2 * jnp.pi * frequencies * dt
#
#     # Limit theta to ensure stability (max π/2 for reasonable behavior)
#     theta = jnp.minimum(theta, jnp.pi / 2)
#
#     # Calculate required c values
#     c = theta * dx / (dt * jnp.pi)
#
#     # Set damping with reduced variability
#     target_magnitude = 1.0 - damping_factor
#     base_k = (1.0 / target_magnitude - 1.0) / dt
#
#     # Reduce variability in damping parameters
#     k_variability = 0.05  # Reduced from 0.2
#
#     # Use more uniform distribution for damping
#     k_random = 1.0 + k_variability * (jax.random.uniform(keys[1], (grid_size,)) * 2 - 1)
#     k = base_k * k_random
#
#     kp_random = 1.0 + k_variability * (jax.random.uniform(keys[2], (grid_size,)) * 2 - 1)
#     kp = base_k * kp_random
#
#     # Reshape to grid format
#     c = c.reshape(grid_side, grid_side)
#     k = k.reshape(grid_side, grid_side)
#     kp = kp.reshape(grid_side, grid_side)
#
#     return c, kp, k

def smooth_spatial_parameters(c, kp, k, smoothness_factor=0.05):
    """
    Apply spatial smoothing to parameters using JAX operations.

    Args:
        c: Wave speed matrix
        kp: Pressure damping matrix
        k: Velocity damping matrix
        smoothness_factor: Controls the smoothing strength

    Returns:
        Smoothed versions of c, kp, and k
    """
    # import jax
    # import jax.numpy as jnp

    def gaussian_kernel_1d(sigma, kernel_size=None):
        """Create a 1D Gaussian kernel."""
        if kernel_size is None:
            # Make the kernel size proportional to sigma
            kernel_size = max(3, int(2 * 3 * sigma + 1))
            # Make sure kernel_size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1

        x = jnp.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        kernel = jnp.exp(-0.5 * (x / sigma) ** 2)
        return kernel / jnp.sum(kernel)

    def apply_gaussian_smoothing(x, sigma):
        """Apply Gaussian smoothing to a 2D array using separable 1D convolutions."""
        # Get the kernel
        kernel = gaussian_kernel_1d(sigma)

        # Apply along rows
        x_rows = jax.vmap(
            lambda row: jnp.convolve(row, kernel, mode='same')
        )(x)

        # Apply along columns
        x_smooth = jax.vmap(
            lambda col: jnp.convolve(col, kernel, mode='same')
        )(x_rows.T).T

        return x_smooth

    # Calculate sigma based on the matrix shape and smoothness factor
    sigma = smoothness_factor * min(c.shape)

    # Apply smoothing to each parameter matrix
    c_smooth = apply_gaussian_smoothing(c, sigma)
    kp_smooth = apply_gaussian_smoothing(kp, sigma)
    k_smooth = apply_gaussian_smoothing(k, sigma)

    return c_smooth, kp_smooth, k_smooth


# def frequency_targeted_regularization(model, alpha=0.01):
#     """Add regularization that penalizes extreme c, kp, k values."""
#     c_reg = alpha * jnp.mean(jnp.square(model.c))
#     kp_reg = alpha * jnp.mean(jnp.square(model.kp))
#     k_reg = alpha * jnp.mean(jnp.square(model.k))
#
#     return c_reg + kp_reg + k_reg
#
#
# def initialize_for_frequency_bands(grid_side, dt, dx, bands, key, variability_factor=0.1):
#     """
#     Initialize parameters to target specific frequency bands with reduced variability.
#
#     Args:
#         grid_side: Side length of the grid
#         dt: Time step
#         dx: Grid spacing
#         bands: List of (min_freq, max_freq, proportion) tuples
#         key: JAX random key
#         variability_factor: Controls the amount of random variability (lower = less specialized)
#     """
#     grid_size = grid_side * grid_side
#     keys = jax.random.split(key, len(bands) + 1)  # Extra key for global noise
#
#     # Create base parameter values with reduced variability
#     c = jnp.zeros((grid_size,))
#     kp = jnp.zeros((grid_size,))
#     k = jnp.zeros((grid_size,))
#
#     # Calculate average values for the entire grid
#     avg_min_freq = sum(band[0] for band in bands) / len(bands)
#     avg_max_freq = sum(band[1] for band in bands) / len(bands)
#
#     # Add a small base value to all parameters for better generalization
#     base_c, base_kp, base_k = initialize_c_and_k(
#         grid_side, dt, dx,
#         avg_min_freq, avg_max_freq,
#         0.05,  # Reduced damping factor
#         keys[0]
#     )
#
#     # Flatten base values
#     base_c = base_c.flatten()
#     base_kp = base_kp.flatten()
#     base_k = base_k.flatten()
#
#     # Mix band-specific and base parameters to reduce specialization
#     start_idx = 0
#     for i, (min_freq, max_freq, proportion) in enumerate(bands):
#         # Calculate how many elements to allocate to this band
#         segment_size = int(grid_size * proportion)
#         end_idx = min(start_idx + segment_size, grid_size)
#         actual_segment_size = end_idx - start_idx
#
#         if actual_segment_size <= 0:
#             continue
#
#         # Calculate a grid size that's a perfect square and smaller than actual_segment_size
#         segment_grid_side = int(jnp.sqrt(actual_segment_size))
#         if segment_grid_side < 1:
#             segment_grid_side = 1
#
#         # Make frequency range less extreme
#         adjusted_min_freq = min_freq * 0.7 + avg_min_freq * 0.3
#         adjusted_max_freq = max_freq * 0.7 + avg_max_freq * 0.3
#
#         # Initialize this segment for the specific band
#         c_segment, kp_segment, k_segment = initialize_c_and_k(
#             segment_grid_side, dt, dx,
#             adjusted_min_freq, adjusted_max_freq,
#             0.05,  # Reduced damping factor
#             keys[i + 1]
#         )
#
#         # Flatten segments
#         c_segment_flat = c_segment.flatten()
#         kp_segment_flat = kp_segment.flatten()
#         k_segment_flat = k_segment.flatten()
#
#         # If segment is too small, tile it to fill the space
#         if c_segment_flat.size < actual_segment_size:
#             repeats = int(jnp.ceil(actual_segment_size / c_segment_flat.size))
#             c_segment_flat = jnp.tile(c_segment_flat, repeats)[:actual_segment_size]
#             kp_segment_flat = jnp.tile(kp_segment_flat, repeats)[:actual_segment_size]
#             k_segment_flat = jnp.tile(k_segment_flat, repeats)[:actual_segment_size]
#         else:
#             # Otherwise just take the required elements
#             c_segment_flat = c_segment_flat[:actual_segment_size]
#             kp_segment_flat = kp_segment_flat[:actual_segment_size]
#             k_segment_flat = k_segment_flat[:actual_segment_size]
#
#         # Mix band-specific values with base values for reduced specialization
#         c_segment_flat = (1 - variability_factor) * base_c[start_idx:end_idx] + variability_factor * c_segment_flat
#         kp_segment_flat = (1 - variability_factor) * base_kp[start_idx:end_idx] + variability_factor * kp_segment_flat
#         k_segment_flat = (1 - variability_factor) * base_k[start_idx:end_idx] + variability_factor * k_segment_flat
#
#         # Assign
#         c = c.at[start_idx:end_idx].set(c_segment_flat)
#         kp = kp.at[start_idx:end_idx].set(kp_segment_flat)
#         k = k.at[start_idx:end_idx].set(k_segment_flat)
#
#         start_idx = end_idx
#         if start_idx >= grid_size:
#             break
#
#     # Apply stronger smoothing to reduce sharp transitions
#     c_reshaped = c.reshape(grid_side, grid_side)
#     kp_reshaped = kp.reshape(grid_side, grid_side)
#     k_reshaped = k.reshape(grid_side, grid_side)
#
#     # Increase smoothness factor for better generalization
#     c_smooth, kp_smooth, k_smooth = smooth_spatial_parameters(
#         c_reshaped, kp_reshaped, k_reshaped,
#         smoothness_factor=0.15  # Increased from default 0.05
#     )
#
#     return c_smooth, kp_smooth, k_smooth

# def fdtd_regularization(model, alpha=0.001):
#     """
#     Add L2 regularization to all FDTD parameters to prevent overfitting.
#
#     Args:
#         model: The FDTD2D model
#         alpha: Regularization strength
#
#     Returns:
#         Regularization loss term
#     """
#     reg_loss = 0.0
#
#     # Add regularization for each block's parameters
#     for block in model.blocks:
#         # Regularize FDTD layer parameters
#         reg_loss += alpha * jnp.sum(jnp.square(block.fdtd.c))
#         reg_loss += alpha * jnp.sum(jnp.square(block.fdtd.kp))
#         reg_loss += alpha * jnp.sum(jnp.square(block.fdtd.k))
#         reg_loss += alpha * jnp.sum(jnp.square(block.fdtd.B))
#         reg_loss += alpha * jnp.sum(jnp.square(block.fdtd.C))
#         reg_loss += alpha * jnp.sum(jnp.square(block.fdtd.D))
#
#         # Regularize GLU weights
#         reg_loss += alpha * sum(jnp.sum(jnp.square(p)) for p in eqx.filter(block.glu, eqx.is_array))
#
#     # Regularize encoder and output layer
#     reg_loss += alpha * sum(jnp.sum(jnp.square(p)) for p in eqx.filter(model.linear_encoder, eqx.is_array))
#     reg_loss += alpha * sum(jnp.sum(jnp.square(p)) for p in eqx.filter(model.linear_layer, eqx.is_array))
#
#     return reg_loss


# def multi_resolution_fdtd_init(grid_side, dt, dx, key):
#     """
#     Initialize FDTD parameters using a multi-resolution approach.
#     Ensures kp and k are diagonal matrices for efficient inversion.
#
#     Args:
#         grid_side: Side length of square grid
#         dt: Time step
#         dx: Grid spacing
#         key: JAX random key
#
#     Returns:
#         c, kp, k: Wave speed and damping parameters for different frequency bands
#     """
#     keys = jr.split(key, 6)
#     grid_size = grid_side * grid_side
#
#     # Create empty grid for parameters
#     c = jnp.zeros((grid_side, grid_side))
#
#     # For kp and k, we'll work with scalar values per grid point
#     # These will form the diagonal of the implicit matrices
#     kp = jnp.zeros((grid_side, grid_side))
#     k = jnp.zeros((grid_side, grid_side))
#
#     # Define quadrant sizes
#     q = grid_side // 2
#
#     # Initialize wave speed (c) for different frequency bands in different quadrants
#     # Top-left: Low frequencies (1-5 Hz) - higher c values
#     c_q1 = 5.0 + 1.0 * jr.normal(keys[0], (q, q))
#
#     # Top-right: Mid-low frequencies (5-15 Hz)
#     c_q2 = 3.0 + 0.5 * jr.normal(keys[1], (q, q))
#
#     # Bottom-left: Mid-high frequencies (15-30 Hz)
#     c_q3 = 1.5 + 0.3 * jr.normal(keys[2], (q, q))
#
#     # Bottom-right: High frequencies (30-50 Hz) - lower c values
#     c_q4 = 0.7 + 0.2 * jr.normal(keys[3], (q, q))
#
#     # Assign to grid with appropriate handling for odd-sized grids
#     c = c.at[:q, :q].set(c_q1)
#     c = c.at[:q, -q:].set(c_q2)
#     c = c.at[-q:, :q].set(c_q3)
#     c = c.at[-q:, -q:].set(c_q4)
#
#     # Initialize damping - use scalar values that will form diagonal matrices
#     # Use lower damping initially to allow model to learn
#     kp_base = 0.005
#     k_base = 0.005
#
#     # Damping follows a similar pattern to c, but at different scales
#     # Each value will be a diagonal element in the damping matrices
#     kp_q1 = kp_base + 0.001 * jr.normal(keys[4], (q, q))
#     kp_q2 = kp_base * 1.2 + 0.001 * jr.normal(keys[4], (q, q))
#     kp_q3 = kp_base * 1.5 + 0.001 * jr.normal(keys[4], (q, q))
#     kp_q4 = kp_base * 2.0 + 0.001 * jr.normal(keys[4], (q, q))
#
#     # Assign to grid
#     kp = kp.at[:q, :q].set(kp_q1)
#     kp = kp.at[:q, -q:].set(kp_q2)
#     kp = kp.at[-q:, :q].set(kp_q3)
#     kp = kp.at[-q:, -q:].set(kp_q4)
#
#     # Same pattern for k
#     k_q1 = k_base + 0.001 * jr.normal(keys[5], (q, q))
#     k_q2 = k_base * 1.2 + 0.001 * jr.normal(keys[5], (q, q))
#     k_q3 = k_base * 1.5 + 0.001 * jr.normal(keys[5], (q, q))
#     k_q4 = k_base * 2.0 + 0.001 * jr.normal(keys[5], (q, q))
#
#     # Assign to grid
#     k = k.at[:q, :q].set(k_q1)
#     k = k.at[:q, -q:].set(k_q2)
#     k = k.at[-q:, :q].set(k_q3)
#     k = k.at[-q:, -q:].set(k_q4)
#
#     # Apply smoothing at the boundaries to avoid sharp transitions
#     c, kp, k = smooth_spatial_parameters(c, kp, k, smoothness_factor=0.1)
#
#     # Ensure CFL stability condition
#     max_c = 0.7 * dx / (dt * jnp.sqrt(2.0))
#     c = jnp.minimum(c, max_c)
#
#     # Ensure positive damping (important for stability)
#     kp = jnp.maximum(kp, 0.001)
#     k = jnp.maximum(k, 0.001)
#
#     return c, kp, k


def multi_resolution_fdtd_init(grid_side, dt, dx, key):
    """
    Initialize FDTD parameters based on dataset frequency analysis.
    """
    keys = jr.split(key, 6)
    grid_size = grid_side * grid_side

    # Create indices for spatial locations
    indices = jnp.arange(grid_size)
    x_indices = indices % grid_side
    y_indices = indices // grid_side

    # Define regions for different frequency bands
    q = grid_side // 2
    is_q1 = (x_indices < q) & (y_indices < q)
    is_q2 = (x_indices >= q) & (y_indices < q)
    is_q3 = (x_indices < q) & (y_indices >= q)
    is_q4 = (x_indices >= q) & (y_indices >= q)

    # Initialize damping parameters (keep low to allow learning)
    kp_base = 0.01
    k_base = 0.01

    kp = jnp.zeros(grid_size)
    k = jnp.zeros(grid_size)

    # Set damping values
    kp = jnp.where(is_q1, kp_base + 0.001 * jr.normal(keys[0], (grid_size,)), kp)
    kp = jnp.where(is_q2, kp_base + 0.001 * jr.normal(keys[0], (grid_size,)), kp)
    kp = jnp.where(is_q3, kp_base + 0.001 * jr.normal(keys[0], (grid_size,)), kp)
    kp = jnp.where(is_q4, kp_base + 0.001 * jr.normal(keys[0], (grid_size,)), kp)

    k = jnp.where(is_q1, k_base + 0.001 * jr.normal(keys[1], (grid_size,)), k)
    k = jnp.where(is_q2, k_base + 0.001 * jr.normal(keys[1], (grid_size,)), k)
    k = jnp.where(is_q3, k_base + 0.001 * jr.normal(keys[1], (grid_size,)), k)
    k = jnp.where(is_q4, k_base + 0.001 * jr.normal(keys[1], (grid_size,)), k)

    # Ensure positive damping
    kp = jnp.maximum(kp, 0.001)
    k = jnp.maximum(k, 0.001)

    # Set target frequencies based on dataset analysis
    # Cover the full range of 0-59 Hz with some emphasis on lower frequencies
    # which are often more important in natural signals
    f_target = jnp.zeros(grid_size)

    # Distribute frequencies across the range 1-59 Hz with more neurons for lower frequencies
    f_low = 1.0 + 4.0 * jr.uniform(keys[2], (grid_size,))  # 1-5 Hz
    f_mid_low = 5.0 + 10.0 * jr.uniform(keys[3], (grid_size,))  # 5-15 Hz
    f_mid_high = 15.0 + 20.0 * jr.uniform(keys[4], (grid_size,))  # 15-35 Hz
    f_high = 35.0 + 24.0 * jr.uniform(keys[5], (grid_size,))  # 35-59 Hz

    # Assign frequencies to different regions
    f_target = jnp.where(is_q1, f_low, f_target)  # 40% for lower frequencies
    f_target = jnp.where(is_q2, f_mid_low, f_target)  # 20% for mid-low frequencies
    f_target = jnp.where(is_q3, f_mid_high, f_target)  # 20% for mid-high frequencies
    f_target = jnp.where(is_q4, f_high, f_target)  # 20% for high frequencies

    # Calculate spatial frequency and wave speeds using eigenvalue relationship
    xi_magnitude = jnp.pi / dx / jnp.sqrt(2)

    # Apply frequency-eigenvalue mapping formula from the paper
    angle = 2 * jnp.pi * f_target * dt

    # Constrain angles for stability
    angle = jnp.minimum(angle, 0.8 * jnp.pi / 2)

    # Calculate wave speed based on target frequency
    numerator = jnp.tan(angle) * jnp.sqrt((1 + dt * kp) * (1 + dt * k))
    denominator = dt * xi_magnitude

    c = numerator / denominator

    # Apply CFL stability condition
    max_c = 0.7 * dx / (dt * jnp.sqrt(2.0))
    c = jnp.minimum(c, max_c)

    # Apply spatial smoothing for better generalization
    c_2d = c.reshape(grid_side, grid_side)

    # Apply custom smoothing
    def gaussian_smooth(x, sigma=0.1 * grid_side):
        kernel_size = max(3, int(2 * 3 * sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1

        x_range = jnp.arange(-(kernel_size // 2), kernel_size // 2 + 1)
        kernel = jnp.exp(-0.5 * (x_range / sigma) ** 2)
        kernel = kernel / jnp.sum(kernel)

        # Apply along rows
        def smooth_row(row):
            return jnp.convolve(row, kernel, mode='same')

        x_rows = jax.vmap(smooth_row)(x)

        # Apply along columns
        def smooth_col(col):
            return jnp.convolve(col, kernel, mode='same')

        return jax.vmap(smooth_col)(x_rows.T).T

    c_2d = gaussian_smooth(c_2d)
    c = c_2d.flatten()

    return c, kp, k

class FDTD2DLayer(eqx.Module):
    """
    FDTD layer implementing the paper's equations (10)-(14).
    """
    c: jax.Array  # wave speed
    kp: jax.Array  # pressure damping
    k: jax.Array  # velocity damping
    dt: float  # time step
    dx: float  # grid spacing
    B: jax.Array  # input matrix
    C: jax.Array  # output matrix
    D: jax.Array  # direct input-to-output matrix
    grid_side: int  # grid dimension
    use_scan: bool  # whether to use associative scan or direct method

    def __init__(self, grid_side, H, *, key, use_scan=True):
        keys = jr.split(key, 6)
        grid_size = grid_side * grid_side


        self.dt = 3.0
        self.dx = 1.0
        self.grid_side = grid_side
        self.use_scan = use_scan
        max_freq = 100.0
        min_freq = 0.0

        # 初始化阻尼系数
        self.kp = jnp.ones((grid_size,)) * 0.01
        self.k = jnp.ones((grid_size,)) * 0.01

        # 生成0到100Hz的随机频率分布
        target_frequency = min_freq + jr.uniform(keys[0], (grid_size,)) * (max_freq - min_freq)

        # 使用中间范围空间频率
        xi_x = jnp.pi / (2 * self.dx)
        xi_y = jnp.pi / (2 * self.dx)
        xi_magnitude = jnp.sqrt(xi_x ** 2 + xi_y ** 2)

        # 针对随机频率计算波速 - 注意需要调整dt以避免数值问题
        # 对于高频，我们需要确保dt足够小
        adjusted_dt = min(self.dt, 0.5 / max_freq)  # 确保dt至少是最高频率周期的一半

        # 使用公式57计算c_{i,j}，但需要注意tan函数周期性
        # 我们确保参数在合理范围内避免数值问题
        angle = 2 * jnp.pi * target_frequency * adjusted_dt
        # 将角度限制在-π/2到π/2范围内避免tan爆炸
        safe_angle = jnp.where(angle > jnp.pi / 4, jnp.pi / 4, angle)

        numerator = jnp.tan(safe_angle) * jnp.sqrt((1 + adjusted_dt * self.kp) * (1 + adjusted_dt * self.k))
        denominator = adjusted_dt * xi_magnitude
        c_init = numerator / denominator

        # 应用CFL稳定性条件
        stability_factor = 0.7
        max_c = stability_factor * (self.dx / adjusted_dt) * jnp.sqrt(
            (1 + adjusted_dt * self.kp) * (1 + adjusted_dt * self.k)) / jnp.sqrt(2.0)

        # 确保波速在合理范围内
        self.c = jnp.minimum(c_init, max_c)
        self.c = jnp.maximum(self.c, 0.01)  # 设置最小波速以避免过慢传播


        # # Initialize parameters
        # self.dt = 1.0
        # # self.dt = jr.uniform(keys[0], minval=1.0, maxval=4.0)
        #
        # self.dx = 1.0
        # self.grid_side = grid_side
        # self.use_scan = use_scan
        #
        # target_freq = 100.0
        # # 初始化阻尼系数
        # self.kp = jnp.ones((grid_size,)) * 0.01  # 压力场阻尼
        # self.k = jnp.ones((grid_size,)) * 0.01  # 速度场阻尼
        #
        # # 使用中间范围空间频率
        # xi_x = jnp.pi / (2 * self.dx)
        # xi_y = jnp.pi / (2 * self.dx)
        # xi_magnitude = jnp.sqrt(xi_x ** 2 + xi_y ** 2)
        #
        # # 使用公式57计算c_{i,j}
        # numerator = jnp.tan(2 * jnp.pi * target_freq * self.dt) * jnp.sqrt(
        #     (1 + self.dt * self.kp) * (1 + self.dt * self.k))
        # denominator = self.dt * xi_magnitude
        # c_init = numerator / denominator
        #
        # # 应用CFL稳定性条件
        # stability_factor = 0.7  # 安全系数
        # max_c = stability_factor * (self.dx / self.dt) * jnp.sqrt(
        #     (1 + self.dt * self.kp) * (1 + self.dt * self.k)) / jnp.sqrt(2.0)
        # self.c = jnp.minimum(c_init, max_c)

        # # # Wave speed (with CFL stability)
        # stddev = 0.001
        # c_init = jnp.abs(jr.normal(keys[0], (grid_size,)) * stddev) + 5.0
        # max_c = 0.7 * (self.dx / (self.dt * jnp.sqrt(2.0)))
        # self.c = jnp.minimum(c_init, max_c)
        #
        # # Damping coefficients
        # self.kp = jnp.abs(jr.normal(keys[1], (grid_size,)) * stddev * 0.0001)
        # self.k = jnp.abs(jr.normal(keys[2], (grid_size,)) * stddev * 0.0001)

        # I/O matrices
        self.B = jr.normal(keys[3], (grid_size, H)) * 0.01
        self.C = jr.normal(keys[4], (H, grid_size)) * 0.01
        self.D = jr.normal(keys[5], (H,)) * 0.01

    # def __init__(self, grid_side, H, *, key, use_scan=True):
    #     keys = jr.split(key, 4)
    #     grid_size = grid_side * grid_side
    #
    #     # Initialize parameters
    #     self.dt = jr.uniform(keys[0], minval=1.0, maxval=3.0)
    #     # self.dt = 1.0  # Time step (can be made trainable)
    #     self.dx = 1.0
    #     self.grid_side = grid_side
    #     self.use_scan = use_scan
    #
    #     # Use multi-resolution initialization
    #     c_init, kp_init, k_init = multi_resolution_fdtd_init(
    #         grid_side, self.dt, self.dx, keys[0]
    #     )
    #
    #     # Flatten arrays for storage
    #     self.c = c_init.flatten()
    #     self.kp = kp_init.flatten()
    #     self.k = k_init.flatten()
    #
    #     # Scale down I/O matrices for better initial gradients
    #     self.B = jr.normal(keys[1], (grid_size, H)) * 0.005
    #     self.C = jr.normal(keys[2], (H, grid_size)) * 0.005
    #     self.D = jr.normal(keys[3], (H,)) * 0.005

    # def __init__(self, grid_side, H, *, key, use_scan=True):
    #     keys = jr.split(key, 6)
    #     grid_size = grid_side * grid_side
    #
    #     # Initialize parameters
    #     self.dt = 1.0
    #     self.dx = 1.0
    #     self.grid_side = grid_side
    #     self.use_scan = use_scan
    #
    #     # Define target frequency bands with less extreme ranges
    #     neural_bands = [
    #         (2, 4, 0.2),  # Lower delta: 2-4 Hz
    #         (4, 7, 0.2),  # Theta: 4-7 Hz
    #         (8, 11, 0.2),  # Lower alpha: 8-11 Hz
    #         (12, 20, 0.2),  # Lower beta: 12-20 Hz
    #         (20, 40, 0.2)  # Lower gamma: 20-40 Hz
    #     ]
    #
    #     # Use reduced variability in initialization
    #     c_init, kp_init, k_init = initialize_for_frequency_bands(
    #         grid_side, self.dt, self.dx, neural_bands, keys[0],
    #         variability_factor=0.1  # Low variability factor
    #     )
    #
    #     # Save the parameters with proper flattening
    #     self.c = c_init.flatten()
    #     self.kp = kp_init.flatten()
    #     self.k = k_init.flatten()
    #
    #     # Reduce the scale of I/O matrices for better generalization
    #     self.B = jr.normal(keys[3], (grid_size, H)) * 0.005  # Reduced from 0.01
    #     self.C = jr.normal(keys[4], (H, grid_size)) * 0.005  # Reduced from 0.01
    #     self.D = jr.normal(keys[5], (H,)) * 0.005  # Reduced from 0.01

    def __call__(self, input_sequence):
        # # Constrain dt to be between 1 and 4
        # constrained_dt = jnp.clip(self.dt, 1.0, 4.0)
        #
        # # Transform input through B matrix
        # Bu = jax.vmap(lambda u: self.B @ u)(input_sequence)
        #
        # # Apply FDTD propagation with constrained dt
        # if self.use_scan:
        #     p, ox, oy = apply_fdtd_2d_scan(
        #         self.c, self.kp, self.k, constrained_dt, self.dx, Bu, self.grid_side
        #     )
        # else:
        #     p, ox, oy = apply_fdtd_2d_direct(
        #         self.c, self.kp, self.k, constrained_dt, self.dx, Bu, self.grid_side
        #     )

        # Transform input through B matrix
        Bu = jax.vmap(lambda u: self.B @ u)(input_sequence)

        # Apply FDTD propagation
        if self.use_scan:
            p, ox, oy = apply_fdtd_2d_scan(
                self.c, self.kp, self.k, self.dt, self.dx, Bu, self.grid_side
            )
        else:
            p, ox, oy = apply_fdtd_2d_direct(
                self.c, self.kp, self.k, self.dt, self.dx, Bu, self.grid_side
            )

        # Output equation (equation 14)
        ys = jax.vmap(lambda x: self.C @ x)(p)
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)


        return ys + Du


class GLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim, output_dim, key):
        w1_key, w2_key = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w1_key)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=w2_key)

    def __call__(self, x):
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


class FDTD2DBlock(eqx.Module):
    norm: eqx.nn.BatchNorm
    fdtd: FDTD2DLayer
    glu: GLU
    drop: eqx.nn.Dropout

    def __init__(
            self,
            grid_side,
            H,
            drop_rate=0.05,
            *,
            key
    ):
        fdtd_key, glu_key = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=H, axis_name="batch", channelwise_affine=False
        )
        self.fdtd = FDTD2DLayer(
            grid_side,
            H,
            key=fdtd_key,
        )
        self.glu = GLU(H, H, key=glu_key)
        self.drop = eqx.nn.Dropout(p=drop_rate)

    def __call__(self, x, state, *, key):
        """Compute FDTD block."""
        dropkey1, dropkey2 = jr.split(key, 2)
        skip = x

        # Apply BatchNorm
        x, state = self.norm(x.T, state)
        x = x.T

        # Apply FDTD
        x = self.fdtd(x)

        # Clip values to prevent extreme outputs
        x = jnp.clip(x, -100.0, 100.0)

        # Apply GLU with dropout
        x = self.drop(jax.nn.gelu(x), key=dropkey1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=dropkey2)

        # Residual connection
        x = skip + x

        return x, state


class FDTD2D(eqx.Module):
    linear_encoder: eqx.nn.Linear
    blocks: list[FDTD2DBlock]
    linear_layer: eqx.nn.Linear
    classification: bool
    output_step: int
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(
            self,
            num_blocks,
            N,
            grid_side,  # Side length of square grid
            H,
            output_dim,
            classification,
            output_step,
            *,
            key
    ):
        linear_encoder_key, *block_keys, linear_layer_key = jr.split(
            key, num_blocks + 2
        )
        # Input encoder
        self.linear_encoder = eqx.nn.Linear(N, H, key=linear_encoder_key)

        # Create FDTD blocks
        self.blocks = [
            FDTD2DBlock(
                grid_side,
                H,
                key=key,
            )
            for key in block_keys
        ]

        # Output layer
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)

        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        """Compute FDTD network."""
        # Split dropout keys for each block
        dropkeys = jr.split(key, len(self.blocks))

        # Encode input
        x = jax.vmap(self.linear_encoder)(x)

        # Apply all blocks
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
            # Clip after each block for stability
            x = jnp.clip(x, -100.0, 100.0)

        # Output layer depends on task type
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1:: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))

        return x, state