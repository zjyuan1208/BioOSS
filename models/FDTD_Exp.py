import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn
import equinox as eqx
from typing import Tuple
import functools

"""This is the final version used in the paper"""
def construct_fdtd_eigendecomposition(
        c: jnp.ndarray,
        kp_diag: jnp.ndarray,
        k_diag: jnp.ndarray,
        dt: jnp.ndarray,  # Now dt is a vector, one value per grid cell
        dx: float,
        grid_side: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Constructs the eigendecomposition matrices for scan.

    Args:
        c: Wave speed array [grid_size]
        kp_diag: Pressure field damping diagonal [grid_side]
        k_diag: Velocity field damping diagonal [grid_side]
        dt: Time step vector [grid_size]
        dx: Grid spacing
        grid_side: Grid dimension

    Returns:
        Lambda: Diagonal eigenvalue vector
        kp_full: Full pressure damping vector
        factor: Factor for eigenvector construction
    """
    grid_size = grid_side * grid_side
    state_size = 3 * grid_size  # [p, ox, oy]

    # Create full damping coefficient vectors from the diagonal values
    kp_full = jnp.zeros((grid_size,))
    k_full = jnp.zeros((grid_size,))

    # Set non-zero values at diagonal positions
    diag_indices = jnp.arange(0, grid_size, grid_side + 1)[:grid_side]
    kp_full = kp_full.at[diag_indices].set(kp_diag)
    k_full = k_full.at[diag_indices].set(k_diag)

    # Ensure positive damping and clip between 0.0001 and 0.5
    kp_full = jnp.clip(nn.softplus(kp_full), 0.0001, 0.5)
    k_full = jnp.clip(nn.softplus(k_full), 0.0001, 0.5)

    # Calculate spatial frequencies based on grid spacing dx
    xi_magnitude = jnp.pi / (2 * dx)  # Mid-range spatial frequency

    # Apply additional safeguard to dt values to prevent NaN
    dt_safe = jnp.clip(dt, 0.1, 5.0)

    # Ensure numerical stability in denominators
    denom1 = jnp.maximum(1.0 + dt_safe * k_full, 0.1)
    denom2 = jnp.maximum(1.0 + dt_safe * kp_full, 0.1)

    # Calculate eigenvalue components with safeguards
    lambda1 = 1.0 / denom1  # First eigenvalue - real, velocity mode
    real_part = 0.5 * (1.0 / denom2 + 1.0 / denom1)

    # Clip c to ensure stability in imaginary part calculation
    c_safe = jnp.clip(c, 0.01, 1.0)

    # Ensure the square root argument is positive
    sqrt_arg = jnp.maximum(denom1 * denom2, 1e-6)
    imag_part = c_safe * dt_safe * xi_magnitude / jnp.sqrt(sqrt_arg)

    # Clip imag_part to prevent excessive oscillation
    imag_part = jnp.clip(imag_part, -10.0, 10.0)

    # Complex conjugate pair for oscillatory modes
    lambda2 = real_part + 1j * imag_part  # Complex eigenvalue - oscillatory mode
    lambda3 = real_part - 1j * imag_part  # Complex conjugate - oscillatory mode

    # Create Lambda vector (diagonal values) - organize in numerical order
    Lambda = jnp.zeros((state_size,), dtype=jnp.complex64)
    Lambda = Lambda.at[:grid_size].set(lambda1)  # Mode 1: velocity mode (real eigenvalue)
    Lambda = Lambda.at[grid_size:2 * grid_size].set(lambda2)  # Mode 2: oscillatory mode (complex eigenvalue)
    Lambda = Lambda.at[2 * grid_size:].set(lambda3)  # Mode 3: oscillatory mode (complex conjugate)

    # Add a larger epsilon to prevent division by zero
    epsilon = 0.001
    # Add safeguards in factor calculation to prevent NaN
    denom_factor = jnp.maximum(jnp.abs(real_part - lambda1), epsilon)
    factor = 1j * imag_part / denom_factor

    return Lambda, kp_full, factor


def apply_fdtd_eigendecomposition_associative(
        c: jnp.ndarray,
        kp_diag: jnp.ndarray,
        k_diag: jnp.ndarray,
        dt: jnp.ndarray,  # Now dt is a vector, one value per grid cell
        dx: float,
        Bu: jnp.ndarray,
        grid_side: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    FDTD propagation using associative scan with eigendecomposition.

    Args:
        c: Wave speed array
        kp_diag: Pressure field damping diagonal
        k_diag: Velocity field damping diagonal
        dt: Time step vector
        dx: Grid spacing
        Bu: Input forcing [seq_len, grid_size]
        grid_side: Grid dimension

    Returns:
        p, ox, oy: Fields over time
    """
    # Ensure Bu has correct shape
    if len(Bu.shape) == 1:
        Bu = Bu.reshape(1, -1)  # Ensure 2D

    seq_len, grid_size = Bu.shape
    state_size = 3 * grid_size  # p, ox, oy components

    # Use the wave speed parameter directly
    # We assume c has already been clipped in the FDTD2DLayer.__call__ method
    c_clipped = c[:grid_size]

    # Clip damping coefficients to ensure they're within the desired range [0.0001, 0.5]
    # First apply softplus to ensure positivity, then clip the result
    kp_diag_pos = jnp.clip(nn.softplus(kp_diag), 0.0001, 0.5)
    k_diag_pos = jnp.clip(nn.softplus(k_diag), 0.0001, 0.5)

    # Construct eigendecomposition parameters with cell-specific dt
    Lambda, kp_full, factor = construct_fdtd_eigendecomposition(
        c_clipped, kp_diag_pos, k_diag_pos, dt, dx, grid_side
    )

    # Prepare input forcing term with proper scaling
    scaling_factor = 1.0 / (1.0 + dt * kp_full[:grid_size])
    scaled_Bu = Bu * scaling_factor

    # Create modal space input (zero-padded for all components)
    F_modal = jnp.zeros((seq_len, state_size), dtype=jnp.complex64)

    # Set modal components - the input affects all state components through eigenvector relations
    # First grid_size elements: mode 1 (velocity mode)
    # These are zero (not directly excited by external forcing)

    # Next grid_size elements: mode 2 (complex eigenvalue)
    F_modal = F_modal.at[:, grid_size:2 * grid_size].set(scaled_Bu)

    # Last grid_size elements: mode 3 (complex conjugate eigenvalue)
    F_modal = F_modal.at[:, 2 * grid_size:].set(jnp.conjugate(scaled_Bu))

    # Create per-step tuples of (Lambda, F_modal[t])
    lambda_seq = jnp.broadcast_to(Lambda, (seq_len, state_size))
    scan_inputs = (lambda_seq, F_modal)

    # Define associative binary operator: (A1, b1) ⊕ (A2, b2) = (A2 * A1, A2 * b1 + b2)
    def binary_op(left, right):
        A1, b1 = left
        A2, b2 = right
        return A2 * A1, A2 * b1 + b2

    # Initial values: identity for Lambda and 0 for state
    init_A = jnp.ones(state_size, dtype=jnp.complex64)
    init_b = jnp.zeros(state_size, dtype=jnp.complex64)

    # Apply associative scan
    A_all, b_all = jax.lax.associative_scan(binary_op, scan_inputs, (init_A, init_b))

    modal_states = b_all  # since init_state = 0

    # Extract the modal field components
    mode1 = modal_states[:, :grid_size]  # Mode 1: velocity mode
    mode2 = modal_states[:, grid_size:2 * grid_size]  # Mode 2: oscillatory mode
    mode3 = modal_states[:, 2 * grid_size:3 * grid_size]  # Mode 3: conjugate oscillatory mode

    # Transform back from modal to physical space
    # Pressure is the real part of the sum of oscillatory modes
    p = jnp.real(mode2 + mode3)  # Only oscillatory modes contribute to pressure

    # Use the factor parameter to reconstruct velocity from pressure modes
    ox_mode2 = factor * mode2  # Velocity component from first oscillatory mode
    ox_mode3 = jnp.conjugate(factor) * mode3  # Velocity component from conjugate mode

    # Velocity fields combine contributions from all modes
    ox = jnp.real(mode1 + ox_mode2 + ox_mode3)  # All modes contribute to velocity
    oy = jnp.real(mode1 + ox_mode2 + ox_mode3)  # Similar structure for y-component

    return p, ox, oy


def initialize_target_frequencies(grid_size, key):
    """
    Initialize target frequencies for neurons with most falling in 1-40 Hz range.
    Uses a skewed distribution to favor the 1-40 Hz range.

    Args:
        grid_size: Number of neurons
        key: Random key for generation

    Returns:
        Array of target frequencies
    """
    # Create a skewed distribution favoring 1-40 Hz
    # Split the key for different distributions
    k1, k2, k3, k4 = jr.split(key, 4)

    # 70% of neurons in the 1-40 Hz range (distributed with beta distribution)
    n_main = int(0.7 * grid_size)
    # n_main = int(0.8 * grid_size)
    # Generate beta-distributed values in [0,1] that favor lower values
    beta_samples = jr.beta(k1, a=2.0, b=3.0, shape=(n_main,))
    # Scale to 1-40 Hz range
    main_freqs = 1.0 + beta_samples * 39.0

    # 20% of neurons in the 40-80 Hz range
    n_high = int(0.2 * grid_size)
    # n_high = int(0.15 * grid_size)
    high_freqs = 40.0 + jr.uniform(k2, shape=(n_high,)) * 40.0

    # 10% of neurons in the 80-100 Hz range
    n_highest = grid_size - n_main - n_high
    highest_freqs = 80.0 + jr.uniform(k3, shape=(n_highest,)) * 20.0

    # Combine all frequencies
    all_freqs = jnp.concatenate([main_freqs, high_freqs, highest_freqs])

    # Shuffle the frequencies
    all_freqs = jr.permutation(k4, all_freqs)

    return all_freqs


class FDTD2DLayer(eqx.Module):
    """
    FDTD layer implementing eigendecomposition-based approach with learnable dt matrix.
    """
    c: jax.Array  # Wave speed (grid_size elements) - trainable
    kp_diag: jax.Array  # Pressure field damping diagonal (grid_side elements) - trainable
    k_diag: jax.Array  # Velocity field damping diagonal (grid_side elements) - trainable
    dt_matrix: jax.Array  # Learnable timestep matrix (grid_side, grid_side) - trainable
    dx: float  # Grid spacing
    B: jax.Array  # Input matrix
    C: jax.Array  # Output matrix
    D: jax.Array  # Direct input to output matrix
    grid_side: int  # Grid dimension
    initial_freq: jax.Array  # Store initial target frequencies for reference


    def __init__(self, grid_side, H, *, key):
        """
        Initialize FDTD2DLayer with eigendecomposition approach and learnable dt matrix.
        """
        keys = jr.split(key, 6)  # Need an extra key for dt initialization
        grid_size = grid_side * grid_side

        # Fixed parameters
        self.dx = 1.0
        self.grid_side = grid_side

        # Calculate target frequencies with concentration in different bands
        f_target = initialize_target_frequencies(grid_size, key=keys[2])
        self.initial_freq = f_target  # Store initial frequencies for reference
        f_target_2d = f_target.reshape(grid_side, grid_side)

        # ---------------------------------------------------------------------
        # Physics-based parameter calculation to achieve target frequencies
        # ---------------------------------------------------------------------

        # Calculate spatial frequencies based on grid spacing
        xi_magnitude = jnp.pi / (2 * self.dx)  # Mid-range spatial frequency

        # Safe ranges for parameters
        min_c, max_c_base = 0.01, 0.5  # Base wave speed range (will adjust based on dt)
        min_kp, max_kp = 0.005, 0.2  # Damping range below clip threshold
        min_k, max_k = 0.005, 0.2  # Damping range below clip threshold
        # min_kp, max_kp = -0.2, 0.2  # Damping range below clip threshold
        # min_k, max_k = -0.2, 0.2  # Damping range below clip threshold

        # Initialize damping coefficients based on frequency
        # Normalize frequencies to [0,1] range for parameter calculation
        norm_freqs = (f_target - jnp.min(f_target)) / (jnp.max(f_target) - jnp.min(f_target) + 1e-8)
        norm_freqs_2d = norm_freqs.reshape(grid_side, grid_side)

        # Calculate damping values proportional to frequency
        # Higher frequencies get higher damping for stability
        kp_values_2d = min_kp + (max_kp - min_kp) * jnp.sqrt(norm_freqs_2d)
        k_values_2d = min_k + (max_k - min_k) * jnp.sqrt(norm_freqs_2d)

        # Extract diagonal elements for damping parameters
        self.kp_diag = jnp.diag(kp_values_2d)
        self.k_diag = jnp.diag(k_values_2d)

        # Initialize learnable dt matrix
        # For each frequency, calculate an appropriate dt that allows moderate wave speeds

        # Calculate moderate damping factor
        damping_factor = jnp.sqrt((1.0 + 0.1) * (1.0 + 0.1))  # Using moderate damping of 0.1

        # Calculate dt values for each target frequency
        # From rearranging the wave equation: dt = arctan(c * xi_magnitude / damping_factor) / (2π*f)
        # Using a moderate wave speed of 0.3 to solve for dt
        c_base = 0.3

        # Avoid division by zero for very small frequencies
        min_freq = 0.01
        f_target_safe = jnp.maximum(f_target_2d, min_freq)

        # Calculate initial dt values
        dt_initial = jnp.arctan(c_base * xi_magnitude / damping_factor) / (1 * jnp.pi * f_target_safe)

        # Clip dt to reasonable range
        dt_initial = jnp.clip(dt_initial, 0.1, 5.0)

        # Store dt in log space to ensure positivity during training
        self.dt_matrix = jnp.log(dt_initial)

        # Initialize wave speeds with moderate values
        # We'll use values that respect CFL condition for each cell's dt
        c_values_2d = jnp.zeros_like(f_target_2d)

        for i in range(grid_side):
            for j in range(grid_side):
                # Calculate CFL limit for this cell's dt
                cfl_factor = 0.7  # Conservative factor (below 1/sqrt(2) ≈ 0.7071)
                cell_dt = dt_initial[i, j]
                max_c_cell = cfl_factor * self.dx / cell_dt

                # Calculate a wave speed that achieves the target frequency
                f = f_target_2d[i, j]
                if f < 0.01:
                    # For very low frequencies, use minimum wave speed
                    c_values_2d = c_values_2d.at[i, j].set(min_c)
                else:
                    # Target wave speed based on formula
                    kp = kp_values_2d[i, j]
                    k = k_values_2d[i, j]
                    damping_factor = jnp.sqrt((1.0 + cell_dt * kp) * (1.0 + cell_dt * k))

                    # Calculate tangent term safely
                    tan_term = jnp.tan(2 * jnp.pi * f * cell_dt)
                    # Check for numerical issues with tangent
                    if jnp.isnan(tan_term) or jnp.abs(tan_term) > 1e3:
                        c_values_2d = c_values_2d.at[i, j].set(max_c_cell * 0.7)
                    else:
                        # Calculate wave speed using the formula
                        c = tan_term * damping_factor / (cell_dt * xi_magnitude)
                        # Ensure it's within CFL limits
                        c = jnp.clip(c, min_c, max_c_cell * 0.8)
                        c_values_2d = c_values_2d.at[i, j].set(c)

        # Apply spatial smoothing to reduce sharp transitions
        c_smooth = jnp.zeros_like(c_values_2d)
        for i in range(-1, 2):
            for j in range(-1, 2):
                c_smooth = c_smooth + jnp.roll(jnp.roll(c_values_2d, i, axis=0), j, axis=1) / 9.0

        # Store wave speed parameter
        self.c = c_smooth.flatten()

        # I/O matrices with appropriate dimensions
        self.B = jr.normal(keys[3], (grid_size, H)) * 0.01  # Input projection
        self.C = jr.normal(keys[4], (H, grid_size)) * 0.01  # Output projection
        self.D = jr.uniform(keys[5], (H,)) * 0.01  # Skip connection


    def get_positive_parameters(self):
        """
        Get positive versions of the parameters with appropriate constraints.
        Uses strict clipping to prevent NaN values during training.

        Returns:
            c_pos: Positive wave speed
            kp_diag_pos: Positive pressure damping
            k_diag_pos: Positive velocity damping
            dt_pos: Positive time step vector
        """
        # Calculate CFL stability limit
        min_c = 0.01

        # For wave speed, use strict clipping to ensure stability
        # First, ensure positivity with softplus, then clip to safe range
        c_pos = nn.softplus(self.c)

        # For damping parameters, ensure positivity with strict clipping
        kp_diag_pos = jnp.clip(nn.softplus(self.kp_diag), 0.0001, 0.2)
        k_diag_pos = jnp.clip(nn.softplus(self.k_diag), 0.0001, 0.2)
        # kp_diag_pos = jnp.clip(self.kp_diag, -0.5, 0.5)
        # k_diag_pos = jnp.clip(self.k_diag, -0.5, 0.5)

        # For dt, convert from log space and apply strict clipping
        # Use narrower bounds for dt to avoid numerical issues
        dt_matrix_pos = jnp.exp(self.dt_matrix)
        dt_matrix_pos = jnp.clip(dt_matrix_pos, 0.1, 5.0)
        # dt_matrix_pos = jnp.clip(self.dt_matrix, 0.1, 5.0)

        # Calculate CFL limit for each grid cell and enforce it strictly
        dt_matrix_reshaped = dt_matrix_pos.reshape(self.grid_side, self.grid_side)
        max_c_matrix = 0.7 * self.dx / dt_matrix_reshaped

        # Reshape c for grid-based CFL check
        c_pos_reshaped = c_pos.reshape(self.grid_side, self.grid_side)

        # Apply CFL condition strictly
        c_pos_safe = jnp.clip(c_pos_reshaped, min_c, 0.9 * max_c_matrix)

        # Flatten for use in simulation
        c_pos_final = c_pos_safe.flatten()
        dt_pos = dt_matrix_pos.flatten()

        return c_pos_final, kp_diag_pos, k_diag_pos, dt_pos

    def __call__(self, x):
        """
        Forward pass through the FDTD layer using associative scan.

        Args:
            x: Input tensor of shape [seq_len, hidden_dim]

        Returns:
            Output tensor of shape [seq_len, hidden_dim]
        """
        # Input projection: [seq_len, hidden_dim] @ [hidden_dim, grid_size]^T = [seq_len, grid_size]
        Bu = x @ self.B.T

        # Get positive parameters with appropriate constraints
        c_pos, kp_diag_pos, k_diag_pos, dt_pos = self.get_positive_parameters()

        # Run FDTD simulation with eigendecomposition approach using variable dt
        p, ox, oy = apply_fdtd_eigendecomposition_associative(
            c_pos, kp_diag_pos, k_diag_pos,
            dt_pos, self.dx, Bu, self.grid_side
        )

        # Output projection (using only pressure field)
        y = p @ self.C.T

        # Add direct connection (skip connection)
        d_contribution = x * self.D  # Broadcasting D across seq_len

        # Combine output
        output = y + d_contribution

        return output


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
            key=fdtd_key
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
            grid_side,
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
                key=key
            )
            for key in block_keys
        ]

        # Output layer
        self.linear_layer = eqx.nn.Linear(H, output_dim, key=linear_layer_key)

        self.classification = classification
        self.output_step = output_step

    def __call__(self, x, state, key):
        """Compute FDTD network."""
        # Allocate dropout keys for each block
        dropkeys = jr.split(key, len(self.blocks))

        # Encode input
        x = jax.vmap(self.linear_encoder)(x)

        # Apply all blocks
        for block, key in zip(self.blocks, dropkeys):
            x, state = block(x, state, key=key)
            # Clip after each block to maintain stability
            x = jnp.clip(x, -100.0, 100.0)

        # Output layer depends on task type
        if self.classification:
            x = jnp.mean(x, axis=0)
            x = jax.nn.softmax(self.linear_layer(x), axis=0)
        else:
            x = x[self.output_step - 1:: self.output_step]
            x = jax.nn.tanh(jax.vmap(self.linear_layer)(x))

        return x, state