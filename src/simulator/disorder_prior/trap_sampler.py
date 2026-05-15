"""
Ge/SiGe charge trap disorder field sampler.

Generates samples from the hierarchical prior:
    p(disorder_field) = integral p(field | theta, n) p(n | theta) p(theta) d(theta) dn

where:
    theta = {(r_k, q_k, type_k, p_k, z_k)}_{k=1}^N   static trap geometry
    n     = {n_k} in {0,1}^N                         dynamic occupancies / orientations

Output: 2D numpy array of shape (nx, ny) in mV, mean-subtracted, matching the
format expected by InitialElectroStaticPotential.compute().

Two trap species
----------------
* MONOPOLE (type=0): single charge +/- 1e at the trap position. Sampled
  uniformly across the device. n_k toggles the charge: occupied -> charge
  present, empty -> nothing.
        U_k(x, y) = (q_k * e_C / (4 pi eps0 eps_r)) * (1/r1 - 1/r2)
        r1 = sqrt(rho^2 + (z_carrier - z_k)^2)    real charge -> 2DHG
        r2 = sqrt(rho^2 + (z_carrier + z_k)^2)    image charge -> 2DHG
  Image is the grounded-conductor mirror at z = -z_k carrying -q_k.

* DIPOLE (type=1): two opposite charges +/- 1e separated by an in-plane
  vector p_k. Sampled preferentially near metallic-gate regions. Each end
  has its own image; the in-plane image-dipole moment opposes the real
  dipole, so the far-field falloff is dressed/quadrupolar (correctly
  reproduced by superposing the two image-charge contributions).
  n_k FLIPS THE ORIENTATION rather than toggling on/off:
        n_k = 1 -> dipole moment in nominal +p direction
        n_k = 0 -> dipole moment in -p direction  (sign flip)
  Models two-state interface fluctuators (TLS) at the gate-oxide region.

Units throughout: nm for lengths, elementary charges (e) for charges,
mV for output.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# Physical constants
EPSILON_0        = 8.854e-12    # F/m
ELEMENTARY_CHARGE = 1.602e-19   # C
NM_TO_M          = 1e-9         # m per nm


@dataclass
class TrapSamplerConfig:
    """
    Physical and sampling parameters for the Ge/SiGe charge trap prior.
    All length units in nm, except where noted.
    """
    # Grid
    grid_shape: Tuple[int, int]       # (nx, ny) pixels  must match gate image
    physical_width_nm: float          # physical width of grid in nm

    # Device geometry
    carrier_depth_nm: float = 40.0    # depth of 2DHG below surface
    trap_depth_nm:    float = 20.0    # depth of Ge/SiGe interface
                                      # must satisfy trap_depth < carrier_depth
    monopole_depth_nm: float = 20.0   # z of monopolar fluctuators
    dipole_depth_nm:   float = 12.0   # z of dipolar fluctuators
    monopole_depth_sigma_nm: float = 0.8  # per-trap depth spread around monopole mean
    dipole_depth_sigma_nm: float = 0.8    # per-trap depth spread around dipole mean

    # Material
    epsilon_r: float = 14.85          # relative permittivity of SiGe

    # Trap count prior: N ~ Poisson(total_trap_density), clipped to [min, max]
    n_traps_mean: float = 10.0
    total_trap_density: float = 10.0
    fraction_dipole: float = 0.3
    n_traps_min:  int   = 1
    n_traps_max:  int   = 50

    # Trap charge: each trap is quantised to exactly +1e (donor, ionised) or
    # -1e (acceptor, occupied). donor_fraction sets the probability of +1e
    # at sample-geometry time; the assignment is fixed for the trajectory.
    # 0.5 = balanced donor/acceptor mix, 1.0 = all donors, 0.0 = all acceptors.
    donor_fraction: float = 0.5

    # Occupancy prior: each trap independently occupied w.p. mean_occupancy
    mean_occupancy: float = 0.5
    # Dipole state prior: probability that dipole starts in +p orientation.
    # 0.5 gives unbiased +/- orientation at t=0.
    dipole_state_bias: float = 0.5
    dipole_length_nm: float = 3.0
    gate_bias_strength: float = 4.0
    rate_log_sigma: float = 0.35

    seed: Optional[int] = None


@dataclass
class TrapGeometry:
    """Static trap configuration for one disorder realisation (theta)."""
    positions_nm: np.ndarray        # shape (N, 2), columns (x_nm, y_nm)
    charges_e:    np.ndarray        # shape (N,), signed, units of e
    dipole_vectors_nm: np.ndarray   # shape (N, 2), zero for monopoles
    trap_types:   np.ndarray        # shape (N,), 0=monopole, 1=dipole
    trap_depths_nm: np.ndarray      # shape (N,), per-trap depth
    gate_bias_weights: Optional[np.ndarray] = None  # (nx, ny), used for biased sampling
    n_traps:      int = field(init=False)

    def __post_init__(self):
        assert self.positions_nm.shape[0] == self.charges_e.shape[0]
        assert self.dipole_vectors_nm.shape[0] == self.charges_e.shape[0]
        assert self.trap_types.shape[0] == self.charges_e.shape[0]
        assert self.trap_depths_nm.shape[0] == self.charges_e.shape[0]
        self.n_traps = self.positions_nm.shape[0]


@dataclass
class DisorderSample:
    """One complete disorder sample: geometry + occupancy + computed field."""
    geometry:    TrapGeometry
    occupancies: np.ndarray         # shape (N,), values in {0, 1}
    field_mV:    np.ndarray         # shape (nx, ny), mean-subtracted, mV


class ChargeTrapSampler:
    """
    Samples disorder fields from the hierarchical Ge/SiGe charge trap prior.

    Usage
    -----
    cfg = TrapSamplerConfig(grid_shape=(150, 150), physical_width_nm=880.0)
    sampler = ChargeTrapSampler(cfg)

    sample = sampler.sample()                # single draw
    samples = sampler.sample_batch(100)      # many draws
    """

    def __init__(self, config: TrapSamplerConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)

        nx, ny = config.grid_shape
        dx = config.physical_width_nm / nx
        dy = config.physical_width_nm / ny

        xs = np.arange(nx) * dx + dx / 2   # pixel centres
        ys = np.arange(ny) * dy + dy / 2
        self.grid_x, self.grid_y = np.meshgrid(xs, ys, indexing='ij')
        self.x_centers = xs
        self.y_centers = ys
        self._uniform_xy_prob = np.full(xs.shape[0] * ys.shape[0], 1.0 / (xs.shape[0] * ys.shape[0]))

        self._prefactor_SI = 1.0 / (4 * np.pi * EPSILON_0 * config.epsilon_r)

    @staticmethod
    def _dilate_binary(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
        out = mask.astype(bool)
        for _ in range(iterations):
            out = (
                out
                | np.roll(out, 1, axis=0)
                | np.roll(out, -1, axis=0)
                | np.roll(out, 1, axis=1)
                | np.roll(out, -1, axis=1)
            )
        return out.astype(float)

    def _build_gate_bias_probabilities(self, gate_bias_weights: Optional[np.ndarray]) -> np.ndarray:
        if gate_bias_weights is None:
            return self._uniform_xy_prob
        w = np.array(gate_bias_weights, dtype=np.float64)
        if w.shape != self.grid_x.shape:
            raise ValueError(
                f"gate_bias_weights shape {w.shape} must match grid shape {self.grid_x.shape}"
            )
        w = np.maximum(w, 0.0)
        gate_region = self._dilate_binary(w > 0.0, iterations=3)
        bias = 1.0 + self.config.gate_bias_strength * gate_region
        flat = bias.ravel()
        return flat / np.sum(flat)

    # Geometry sampling

    def sample_geometry(self, gate_bias_weights: Optional[np.ndarray] = None) -> TrapGeometry:
        """Draw one static trap configuration theta ~ p(theta)."""
        cfg = self.config

        N = int(np.clip(
            self.rng.poisson(cfg.total_trap_density),
            cfg.n_traps_min,
            cfg.n_traps_max,
        ))
        n_dipole = int(np.round(cfg.fraction_dipole * N))
        n_dipole = int(np.clip(n_dipole, 0, N))
        n_monopole = N - n_dipole

        gate_prob = self._build_gate_bias_probabilities(gate_bias_weights)
        # Monopoles sampled broadly; dipoles biased near gate regions if provided later.
        mono_idx = self.rng.choice(gate_prob.size, size=n_monopole, replace=True, p=self._uniform_xy_prob)
        dip_idx = self.rng.choice(gate_prob.size, size=n_dipole, replace=True, p=gate_prob)
        all_idx = np.concatenate([mono_idx, dip_idx])

        ix = all_idx // self.grid_x.shape[1]
        iy = all_idx % self.grid_x.shape[1]
        x_positions = self.x_centers[ix]
        y_positions = self.y_centers[iy]
        positions = np.stack([x_positions, y_positions], axis=1)

        # Each trap is quantised: sign = +1 (donor) with probability donor_fraction,
        # else -1 (acceptor). Magnitude is exactly 1e  no fractional charges.
        is_donor = self.rng.random(size=N) < cfg.donor_fraction
        charges  = np.where(is_donor, 1.0, -1.0)   # units of e
        trap_types = np.concatenate([
            np.zeros(n_monopole, dtype=np.int8),
            np.ones(n_dipole, dtype=np.int8),
        ])
        trap_depths_nm = np.zeros(N, dtype=np.float64)
        mono_sigma = max(float(cfg.monopole_depth_sigma_nm), 0.0)
        dip_sigma = max(float(cfg.dipole_depth_sigma_nm), 0.0)
        carrier_limit = cfg.carrier_depth_nm - 1e-3
        if n_monopole > 0:
            trap_depths_nm[:n_monopole] = self.rng.normal(
                loc=cfg.monopole_depth_nm,
                scale=mono_sigma,
                size=n_monopole,
            )
        if n_dipole > 0:
            trap_depths_nm[n_monopole:] = self.rng.normal(
                loc=cfg.dipole_depth_nm,
                scale=dip_sigma,
                size=n_dipole,
            )
        trap_depths_nm = np.clip(trap_depths_nm, 1.0, carrier_limit)

        dipole_vectors = np.zeros((N, 2), dtype=np.float64)
        if n_dipole > 0:
            phi = self.rng.uniform(0.0, 2 * np.pi, size=n_dipole)
            # Mostly in-plane dipoles near gates.
            lengths = np.clip(
                self.rng.lognormal(mean=np.log(max(cfg.dipole_length_nm, 0.1)), sigma=0.25, size=n_dipole),
                0.2,
                10.0,
            )
            dipole_vectors[n_monopole:, 0] = lengths * np.cos(phi)
            dipole_vectors[n_monopole:, 1] = lengths * np.sin(phi)

        return TrapGeometry(
            positions_nm=positions,
            charges_e=charges,
            dipole_vectors_nm=dipole_vectors,
            trap_types=trap_types,
            trap_depths_nm=trap_depths_nm,
            gate_bias_weights=gate_bias_weights,
        )

    # Occupancy sampling

    def sample_occupancies(self, geometry: TrapGeometry) -> np.ndarray:
        """
        Draw initial two-state values per trap.

        - Monopoles (type=0): occupancy state, Bernoulli(mean_occupancy)
        - Dipoles   (type=1): orientation state, Bernoulli(dipole_state_bias)
            1 -> +p orientation, 0 -> -p orientation
        """
        n_traps = geometry.n_traps
        out = np.zeros(n_traps, dtype=np.uint8)
        mono = geometry.trap_types == 0
        dip = ~mono
        if np.any(mono):
            out[mono] = self.rng.binomial(1, self.config.mean_occupancy, size=int(np.sum(mono)))
        if np.any(dip):
            bias = float(np.clip(self.config.dipole_state_bias, 0.0, 1.0))
            out[dip] = self.rng.binomial(1, bias, size=int(np.sum(dip)))
        return out

    # Field computation

    def compute_field(
        self,
        geometry: TrapGeometry,
        occupancies: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the disorder potential field in mV, mean-subtracted.

        Monopole traps contribute only when occupancy==1 (donor ionised /
        acceptor occupied -> charge present).

        Dipole traps are two-state telegraph fluctuators: they ALWAYS
        contribute, but flip orientation with the occupancy state.
            occupancy = 1 -> dipole moment in nominal +p direction
            occupancy = 0 -> dipole moment in -p direction
        Net effect: the field switches sign between the two states rather
        than between zero and a nonzero value, doubling the per-flip
        fluctuation amplitude (matches TLS/two-state-defect physics near
        gate interfaces).
        """
        assert occupancies.shape[0] == geometry.n_traps

        nx, ny = self.config.grid_shape
        out = np.zeros((nx, ny))

        for k in range(geometry.n_traps):
            is_dipole = geometry.trap_types[k] == 1

            if not is_dipole and occupancies[k] != 1:
                # Empty monopole contributes nothing.
                continue

            x_k    = geometry.positions_nm[k, 0]
            y_k    = geometry.positions_nm[k, 1]
            q_k_e  = geometry.charges_e[k]
            z_k_nm = geometry.trap_depths_nm[k]

            q_k_C = q_k_e * ELEMENTARY_CHARGE

            dx_m = (self.grid_x - x_k) * NM_TO_M
            dy_m = (self.grid_y - y_k) * NM_TO_M
            rho2 = dx_m ** 2 + dy_m ** 2

            delta_z_m = (self.config.carrier_depth_nm - z_k_nm) * NM_TO_M
            z_sum_m   = (self.config.carrier_depth_nm + z_k_nm) * NM_TO_M
            r1 = np.sqrt(rho2 + delta_z_m ** 2)   # real charge -> 2DHG
            r2 = np.sqrt(rho2 + z_sum_m ** 2)     # image charge -> 2DHG

            if not is_dipole:
                # Monopole: only here if occupancies[k] == 1 (filtered above)
                U_k_V = self._prefactor_SI * q_k_C * (1.0 / r1 - 1.0 / r2)
                out += U_k_V * 1e3
                continue

            # Dipole: always contributes. Sign of dipole moment flips
            # with occupancy state (orientation-flipping TLS).
            sign = 1.0 if occupancies[k] == 1 else -1.0

            p_k = geometry.dipole_vectors_nm[k]
            x_plus  = x_k + 0.5 * p_k[0]
            y_plus  = y_k + 0.5 * p_k[1]
            x_minus = x_k - 0.5 * p_k[0]
            y_minus = y_k - 0.5 * p_k[1]

            dxp_m = (self.grid_x - x_plus) * NM_TO_M
            dyp_m = (self.grid_y - y_plus) * NM_TO_M
            rhop2 = dxp_m ** 2 + dyp_m ** 2
            rp1 = np.sqrt(rhop2 + delta_z_m ** 2)
            rp2 = np.sqrt(rhop2 + z_sum_m ** 2)

            dxm_m = (self.grid_x - x_minus) * NM_TO_M
            dym_m = (self.grid_y - y_minus) * NM_TO_M
            rhom2 = dxm_m ** 2 + dym_m ** 2
            rm1 = np.sqrt(rhom2 + delta_z_m ** 2)
            rm2 = np.sqrt(rhom2 + z_sum_m ** 2)

            U_plus  = self._prefactor_SI * ( q_k_C) * (1.0 / rp1 - 1.0 / rp2)
            U_minus = self._prefactor_SI * (-q_k_C) * (1.0 / rm1 - 1.0 / rm2)
            out += sign * (U_plus + U_minus) * 1e3

        out -= np.mean(out)   # SiGe mean-subtraction convention
        return out

    # Full draws

    def sample(self) -> DisorderSample:
        """One draw from the full hierarchical prior."""
        geometry    = self.sample_geometry()
        occupancies = self.sample_occupancies(geometry)
        fld         = self.compute_field(geometry, occupancies)
        return DisorderSample(geometry=geometry, occupancies=occupancies, field_mV=fld)

    def sample_batch(self, n_samples: int) -> list:
        """n_samples independent disorder samples."""
        return [self.sample() for _ in range(n_samples)]
