"""
Hyperprior over TrapSamplerConfig parameters.

Encodes epistemic uncertainty about the device's trap distribution: we have
one specific device whose true hyperparameters (n_traps_mean, donor_fraction,
mean_occupancy, ...) are unknown, so the training distribution averages over
plausible values rather than committing to a point estimate.

Usage
-----
    hyperprior = load_hyperprior(yaml.safe_load(open("hyperprior.yaml")))
    sampler_cfg, drawn = draw_sampler_config(
        hyperprior,
        grid_shape=(150, 150),
        physical_width_nm=880.0,
    )
    sampler = ChargeTrapSampler(sampler_cfg)

    # `drawn` is a dict of the actual scalar values realised this draw  record
    # them in the trajectory metadata so the transformer can use them as
    # auxiliary labels (or so you can audit the training set later).
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import numpy as np

from simulator.disorder_prior.trap_sampler import TrapSamplerConfig


# Distributions ---------------------------------------------------------------

@dataclass
class Fixed:
    """Pin a parameter to a constant. Use to lock things you do know."""
    value: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(self.value)


@dataclass
class Uniform:
    low:  float
    high: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.low, self.high))


@dataclass
class Normal:
    mu:    float
    sigma: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.normal(self.mu, self.sigma))


@dataclass
class LogNormal:
    """
    Natural choice for positive-valued physical scales when uncertainty
    spans an order of magnitude. mu and sigma are mean/std of the
    underlying normal in log-space (numpy convention).
    """
    mu:    float
    sigma: float

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.lognormal(self.mu, self.sigma))


@dataclass
class Beta:
    """Beta(alpha, beta) scaled to [0, scale]. Bounded support."""
    alpha: float
    beta:  float
    scale: float = 1.0

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.beta(self.alpha, self.beta) * self.scale)


Distribution = Union[Fixed, Uniform, Normal, LogNormal, Beta]


_DIST_REGISTRY = {
    "fixed":     Fixed,
    "uniform":   Uniform,
    "normal":    Normal,
    "lognormal": LogNormal,
    "beta":      Beta,
}


def load_distribution(spec: dict) -> Distribution:
    """Build a distribution from a dict like {type: lognormal, mu: 2.3, sigma: 0.5}."""
    spec = dict(spec)
    dist_type = spec.pop("type")
    if dist_type not in _DIST_REGISTRY:
        raise ValueError(
            f"Unknown distribution type {dist_type!r}. "
            f"Known: {sorted(_DIST_REGISTRY)}"
        )
    return _DIST_REGISTRY[dist_type](**spec)


# Hyperprior over TrapSamplerConfig -------------------------------------------

@dataclass
class HyperpriorConfig:
    """
    Distributions over the configurable parameters of TrapSamplerConfig.

    Genuinely uncertain quantities for the device:
      - n_traps_mean    Poisson mean for the trap count
      - donor_fraction  fraction of traps that are donor-type (+1e when ionised)
                        vs acceptor-type (-1e when occupied). Each trap is
                        quantised  no fractional charges.
      - mean_occupancy  thermal-equilibrium occupancy fraction
      - trap_depth_nm   where the traps actually sit; multiple plausible
                        interfaces (Ge/SiGe, Si/SiOx, buffer)  genuinely unknown
    Held fixed (set from device fab specs in draw_sampler_config):
      - carrier_depth_nm  ~well-known from heterostructure design
      - epsilon_r         material constant
      - grid_shape, physical_width_nm  pinned by gate layout

    n_traps_min / n_traps_max are hard clips on the Poisson tail.
    """
    n_traps_mean:   Distribution
    donor_fraction: Distribution
    mean_occupancy: Distribution
    total_trap_density: Distribution = field(default_factory=lambda: Fixed(10.0))
    fraction_dipole: Distribution = field(default_factory=lambda: Fixed(0.3))
    trap_depth_nm:  Distribution = field(default_factory=lambda: Fixed(20.0))
    monopole_depth_nm: Distribution = field(default_factory=lambda: Fixed(20.0))
    dipole_depth_nm: Distribution = field(default_factory=lambda: Fixed(12.0))
    monopole_depth_sigma_nm: Distribution = field(default_factory=lambda: Fixed(0.8))
    dipole_depth_sigma_nm: Distribution = field(default_factory=lambda: Fixed(0.8))
    dipole_length_nm: Distribution = field(default_factory=lambda: Fixed(3.0))
    # Strength of the dipole-near-gate sampling bias.
    # bias = 1 + gate_bias_strength * (in_gate_region ? 1 : 0)
    # With 22% gate coverage:
    #   strength=4   -> ~58% of dipoles on gates (default; visually subtle)
    #   strength=20  -> ~85% of dipoles on gates (visually obvious)
    #   strength=50  -> ~93% of dipoles on gates (strong cluster)
    gate_bias_strength: Distribution = field(default_factory=lambda: Fixed(20.0))

    n_traps_min: int = 1
    n_traps_max: int = 50


def load_hyperprior(spec: dict) -> HyperpriorConfig:
    """Build a HyperpriorConfig from a YAML-loaded dict."""
    spec = dict(spec)
    trap_depth_spec = spec.get("trap_depth_nm")
    total_density_spec = spec.get("total_trap_density")
    fraction_dipole_spec = spec.get("fraction_dipole")
    monopole_depth_spec = spec.get("monopole_depth_nm")
    dipole_depth_spec = spec.get("dipole_depth_nm")
    monopole_depth_sigma_spec = spec.get("monopole_depth_sigma_nm")
    dipole_depth_sigma_spec = spec.get("dipole_depth_sigma_nm")
    dipole_length_spec = spec.get("dipole_length_nm")
    gate_bias_spec = spec.get("gate_bias_strength")
    return HyperpriorConfig(
        n_traps_mean   = load_distribution(spec["n_traps_mean"]),
        total_trap_density = (load_distribution(total_density_spec)
                              if total_density_spec is not None
                              else load_distribution(spec["n_traps_mean"])),
        fraction_dipole = (load_distribution(fraction_dipole_spec)
                           if fraction_dipole_spec is not None
                           else Fixed(0.3)),
        donor_fraction = load_distribution(spec["donor_fraction"]),
        mean_occupancy = load_distribution(spec["mean_occupancy"]),
        trap_depth_nm  = (load_distribution(trap_depth_spec)
                          if trap_depth_spec is not None
                          else Fixed(20.0)),
        monopole_depth_nm = (load_distribution(monopole_depth_spec)
                             if monopole_depth_spec is not None
                             else (load_distribution(trap_depth_spec)
                                   if trap_depth_spec is not None
                                   else Fixed(20.0))),
        dipole_depth_nm = (load_distribution(dipole_depth_spec)
                           if dipole_depth_spec is not None
                           else Fixed(12.0)),
        monopole_depth_sigma_nm = (load_distribution(monopole_depth_sigma_spec)
                                   if monopole_depth_sigma_spec is not None
                                   else Fixed(0.8)),
        dipole_depth_sigma_nm = (load_distribution(dipole_depth_sigma_spec)
                                 if dipole_depth_sigma_spec is not None
                                 else Fixed(0.8)),
        dipole_length_nm = (load_distribution(dipole_length_spec)
                            if dipole_length_spec is not None
                            else Fixed(3.0)),
        gate_bias_strength = (load_distribution(gate_bias_spec)
                              if gate_bias_spec is not None
                              else Fixed(20.0)),
        n_traps_min    = int(spec.get("n_traps_min", 1)),
        n_traps_max    = int(spec.get("n_traps_max", 50)),
    )


def draw_sampler_config(
    hyperprior:        HyperpriorConfig,
    grid_shape:        Tuple[int, int],
    physical_width_nm: float,
    carrier_depth_nm:  float = 40.0,
    epsilon_r:         float = 14.85,
    min_delta_z_nm:    float = 1.0,
    rng:               Optional[np.random.Generator] = None,
    seed:              Optional[int] = None,
) -> Tuple[TrapSamplerConfig, dict]:
    """
    Draw one TrapSamplerConfig from the hyperprior.

    Parameters
    ----------
    hyperprior        : prior over trap-population properties + trap_depth_nm
    grid_shape        : (nx, ny) of the gate image  pinned by the device
    physical_width_nm : device width  pinned by the device
    carrier_depth_nm  : 2DHG depth  known from heterostructure design
    epsilon_r         : permittivity  material constant
    min_delta_z_nm    : minimum allowed (carrier_depth - trap_depth); trap
                        depth draws are clipped so this stays positive (the
                        image-charge formula needs trap_depth < carrier_depth)
    rng / seed        : provide either an existing generator or a seed

    Returns
    -------
    sampler_config : TrapSamplerConfig ready for ChargeTrapSampler(...)
    drawn          : dict {param_name: realised_value}  record in metadata
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    # Clip trap depth so it stays at least min_delta_z_nm below the carrier.
    # Without this, Beta/Normal tails can produce trap_depth >= carrier_depth
    # which the image-charge formula can't handle (delta_z must be positive).
    raw_trap_depth = hyperprior.trap_depth_nm.sample(rng)
    trap_depth_nm  = float(np.clip(raw_trap_depth,
                                   1.0,
                                   carrier_depth_nm - min_delta_z_nm))
    raw_monopole_depth = hyperprior.monopole_depth_nm.sample(rng)
    monopole_depth_nm = float(np.clip(raw_monopole_depth,
                                      1.0,
                                      carrier_depth_nm - min_delta_z_nm))
    raw_dipole_depth = hyperprior.dipole_depth_nm.sample(rng)
    dipole_depth_nm = float(np.clip(raw_dipole_depth,
                                    1.0,
                                    carrier_depth_nm - min_delta_z_nm))
    monopole_depth_sigma_nm = float(np.clip(
        hyperprior.monopole_depth_sigma_nm.sample(rng),
        0.0,
        5.0,
    ))
    dipole_depth_sigma_nm = float(np.clip(
        hyperprior.dipole_depth_sigma_nm.sample(rng),
        0.0,
        5.0,
    ))
    dipole_length_nm = float(np.clip(hyperprior.dipole_length_nm.sample(rng), 0.1, 20.0))
    fraction_dipole = float(np.clip(hyperprior.fraction_dipole.sample(rng), 0.0, 1.0))
    gate_bias_strength = float(np.clip(hyperprior.gate_bias_strength.sample(rng), 0.0, 200.0))

    # donor_fraction must be in [0, 1] to be a valid probability  clip
    # tail draws (e.g. Beta with extreme params is fine, but Normal could
    # produce negatives or >1).
    donor_fraction = float(np.clip(hyperprior.donor_fraction.sample(rng), 0.0, 1.0))

    drawn = {
        "n_traps_mean":   hyperprior.n_traps_mean.sample(rng),
        "total_trap_density": hyperprior.total_trap_density.sample(rng),
        "fraction_dipole": fraction_dipole,
        "donor_fraction": donor_fraction,
        "mean_occupancy": hyperprior.mean_occupancy.sample(rng),
        "trap_depth_nm":  trap_depth_nm,
        "monopole_depth_nm": monopole_depth_nm,
        "dipole_depth_nm": dipole_depth_nm,
        "monopole_depth_sigma_nm": monopole_depth_sigma_nm,
        "dipole_depth_sigma_nm": dipole_depth_sigma_nm,
        "dipole_length_nm": dipole_length_nm,
        "gate_bias_strength": gate_bias_strength,
    }

    cfg = TrapSamplerConfig(
        grid_shape         = grid_shape,
        physical_width_nm  = physical_width_nm,
        carrier_depth_nm   = carrier_depth_nm,
        trap_depth_nm      = trap_depth_nm,
        monopole_depth_nm  = monopole_depth_nm,
        dipole_depth_nm    = dipole_depth_nm,
        monopole_depth_sigma_nm = monopole_depth_sigma_nm,
        dipole_depth_sigma_nm = dipole_depth_sigma_nm,
        dipole_length_nm   = dipole_length_nm,
        gate_bias_strength = gate_bias_strength,
        epsilon_r          = epsilon_r,
        n_traps_mean       = drawn["n_traps_mean"],
        total_trap_density = drawn["total_trap_density"],
        fraction_dipole    = drawn["fraction_dipole"],
        n_traps_min        = hyperprior.n_traps_min,
        n_traps_max        = hyperprior.n_traps_max,
        donor_fraction     = drawn["donor_fraction"],
        mean_occupancy     = drawn["mean_occupancy"],
        # Forward an independent seed so the per-trajectory sampler is
        # deterministic given (hyperprior_seed, sample_id)
        seed = int(rng.integers(0, 2**31 - 1)),
    )
    return cfg, drawn
