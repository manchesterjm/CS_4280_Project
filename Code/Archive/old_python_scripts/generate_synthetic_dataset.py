"""
Generate Balanced Synthetic Light Curve Dataset for Exoplanet Detection
Creates 1000 light curves: 500 with planets, 500 without (flares, EB, noise, background events)

Usage:
    conda activate exo-lstm-gpu
    python generate_synthetic_dataset.py --output_dir "C:\CS_4280_Project\synthetic_balanced_dataset" --n_planets 500 --n_non_planets 500 --seed 42

Requirements:
    pip install batman-package  # For transit modeling
    conda install -c conda-forge lightkurve  # For realistic noise
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Try to import batman (transit modeling)
try:
    import batman
except ImportError:
    print("WARNING: batman-package not installed. Install with: pip install batman-package")
    batman = None


class SyntheticLightCurveGenerator:
    """Generate realistic synthetic light curves for training"""

    def __init__(self, duration_days=27.0, cadence_minutes=2.0, noise_ppm=200.0, seed=42):
        """
        Args:
            duration_days: Total observation time (TESS sector ~ 27 days)
            cadence_minutes: Time between observations (TESS 2-min cadence)
            noise_ppm: Photon noise level in parts per million
            seed: Random seed for reproducibility
        """
        self.duration_days = duration_days
        self.cadence_minutes = cadence_minutes
        self.noise_ppm = noise_ppm
        self.seed = seed

        # Generate time array
        cadence_days = cadence_minutes / (60.0 * 24.0)
        self.time = np.arange(0, duration_days, cadence_days)
        self.n_points = len(self.time)

        np.random.seed(seed)

    def add_noise(self, flux):
        """Add realistic TESS-like noise to flux"""
        # Photon noise (Gaussian)
        photon_noise = np.random.normal(0, self.noise_ppm * 1e-6, len(flux))

        # Stellar variability (low-frequency sine waves)
        n_modes = np.random.randint(1, 4)
        stellar_var = np.zeros_like(flux)
        for _ in range(n_modes):
            period = np.random.uniform(1.0, 10.0)  # Days
            amplitude = np.random.uniform(50, 300) * 1e-6  # 50-300 ppm
            phase = np.random.uniform(0, 2*np.pi)
            stellar_var += amplitude * np.sin(2*np.pi*self.time/period + phase)

        # Systematic trend (linear drift)
        trend_slope = np.random.uniform(-50, 50) * 1e-6 / self.duration_days
        trend = trend_slope * (self.time - self.time[0])

        # Combine
        noisy_flux = flux + photon_noise + stellar_var + trend

        return noisy_flux

    def generate_planet_transit(self, tic_id):
        """
        Generate a light curve with planetary transit
        Uses batman for realistic transit modeling

        Returns:
            time, flux, metadata dict
        """
        if batman is None:
            raise ImportError("batman-package required. Install with: pip install batman-package")

        # Random transit parameters (TESS-like)
        period = np.random.uniform(1.0, 30.0)  # Days
        t0 = np.random.uniform(0, period)  # Time of first transit

        # Planet size (Earth to Jupiter: 1-11 Earth radii)
        planet_radius_earth = np.random.uniform(1.0, 11.0)

        # Stellar radius (assume Sun-like: 0.7-1.3 solar radii)
        star_radius_solar = np.random.uniform(0.7, 1.3)

        # Calculate transit depth (Rp/Rs)^2
        # 1 Earth radius = 0.009 Solar radii, 1 Jupiter radius = 0.1 Solar radii
        planet_radius_solar = planet_radius_earth * 0.009155
        rp_over_rs = planet_radius_solar / star_radius_solar

        # Transit depth in flux (should be 0.1-2% for TESS-like)
        depth = rp_over_rs**2

        # Transit duration: 1-6 hours for typical hot planets
        duration_hours = np.random.uniform(1.0, 6.0)
        duration_days = duration_hours / 24.0

        # Orbital parameters
        a_over_rs = ((period / 3.0) ** (2/3)) * 215.0  # Scaled from Kepler's 3rd law
        inclination = np.random.uniform(88.0, 90.0)  # High inclination for transits

        # Limb darkening (quadratic, Sun-like star)
        u1 = np.random.uniform(0.3, 0.5)
        u2 = np.random.uniform(0.1, 0.3)

        # Batman parameters
        params = batman.TransitParams()
        params.t0 = t0                    # Time of inferior conjunction
        params.per = period               # Orbital period (days)
        params.rp = rp_over_rs            # Planet radius (in units of stellar radii)
        params.a = a_over_rs              # Semi-major axis (in units of stellar radii)
        params.inc = inclination          # Orbital inclination (degrees)
        params.ecc = 0.0                  # Eccentricity (circular orbit)
        params.w = 90.0                   # Longitude of periastron (degrees)
        params.u = [u1, u2]               # Limb darkening coefficients
        params.limb_dark = "quadratic"    # Limb darkening model

        # Generate transit model
        m = batman.TransitModel(params, self.time)
        flux_clean = m.light_curve(params)

        # Add noise
        flux = self.add_noise(flux_clean)

        # Metadata
        metadata = {
            'tic_id': tic_id,
            'label': 'planet',
            'period': period,
            'depth': depth,
            'duration': duration_days,
            't0': t0,
            'planet_radius_earth': planet_radius_earth,
            'star_radius_solar': star_radius_solar,
            'inclination': inclination
        }

        return self.time.copy(), flux, metadata

    def generate_stellar_flare(self, tic_id):
        """Generate a light curve with stellar flare(s)"""
        flux = np.ones(self.n_points)

        # Random number of flares (1-5)
        n_flares = np.random.randint(1, 6)

        flare_amplitudes = []
        flare_times = []

        for _ in range(n_flares):
            # Flare parameters
            amplitude = np.random.uniform(0.005, 0.05)  # 0.5-5% brightness increase
            peak_time = np.random.uniform(0.1 * self.duration_days, 0.9 * self.duration_days)
            rise_time = np.random.uniform(0.01, 0.05)  # Days (rapid rise)
            decay_time = np.random.uniform(0.05, 0.3)  # Days (slow decay)

            # Asymmetric flare profile (fast rise, slow decay)
            for i, t in enumerate(self.time):
                dt = t - peak_time
                if dt < 0:
                    # Rising phase (Gaussian)
                    flux[i] += amplitude * np.exp(-(dt**2) / (2 * rise_time**2))
                else:
                    # Decay phase (exponential)
                    flux[i] += amplitude * np.exp(-dt / decay_time)

            flare_amplitudes.append(amplitude)
            flare_times.append(peak_time)

        # Add noise
        flux = self.add_noise(flux)

        metadata = {
            'tic_id': tic_id,
            'label': 'flare',
            'n_flares': n_flares,
            'max_amplitude': max(flare_amplitudes),
            'flare_times': flare_times
        }

        return self.time.copy(), flux, metadata

    def generate_eclipsing_binary(self, tic_id):
        """Generate an eclipsing binary (EB) light curve"""
        flux = np.ones(self.n_points)

        # Binary parameters
        period = np.random.uniform(0.5, 10.0)  # Days (shorter than planets)
        t0 = np.random.uniform(0, period)

        # Primary eclipse (deeper)
        primary_depth = np.random.uniform(0.02, 0.15)  # 2-15% (V-shaped, deeper than planets)
        primary_duration = np.random.uniform(0.05, 0.2)  # Days

        # Secondary eclipse (shallower, phase 0.5)
        secondary_depth = primary_depth * np.random.uniform(0.3, 0.8)
        secondary_duration = primary_duration * np.random.uniform(0.8, 1.2)

        # Generate eclipses
        for i, t in enumerate(self.time):
            phase = ((t - t0) % period) / period

            # Primary eclipse (phase ~ 0)
            if phase < primary_duration / period or phase > (1 - primary_duration / period):
                # V-shaped eclipse (flatter bottom than transits)
                relative_phase = min(phase, 1 - phase) / (primary_duration / period)
                flux[i] -= primary_depth * (1 - relative_phase)

            # Secondary eclipse (phase ~ 0.5)
            elif abs(phase - 0.5) < secondary_duration / (2 * period):
                relative_phase = abs(phase - 0.5) / (secondary_duration / period)
                flux[i] -= secondary_depth * (1 - relative_phase)

        # Add noise
        flux = self.add_noise(flux)

        metadata = {
            'tic_id': tic_id,
            'label': 'EB',
            'period': period,
            'primary_depth': primary_depth,
            'secondary_depth': secondary_depth,
            't0': t0
        }

        return self.time.copy(), flux, metadata

    def generate_pure_noise(self, tic_id):
        """Generate a light curve with only noise (no signal)"""
        flux = np.ones(self.n_points)

        # Add noise
        flux = self.add_noise(flux)

        metadata = {
            'tic_id': tic_id,
            'label': 'noise',
            'noise_ppm': self.noise_ppm
        }

        return self.time.copy(), flux, metadata

    def generate_background_event(self, tic_id):
        """Generate a light curve with background contamination event"""
        flux = np.ones(self.n_points)

        # Random number of events (1-3)
        n_events = np.random.randint(1, 4)

        for _ in range(n_events):
            # Event parameters (sudden jump/drop)
            event_time = np.random.uniform(0.2 * self.duration_days, 0.8 * self.duration_days)
            amplitude = np.random.uniform(-0.03, 0.03)  # ±3% change
            duration = np.random.uniform(0.1, 2.0)  # Days

            # Create event (step function or ramp)
            event_type = np.random.choice(['step', 'ramp'])

            if event_type == 'step':
                # Sudden jump
                flux[self.time > event_time] += amplitude
            else:
                # Gradual ramp
                mask = (self.time > event_time) & (self.time < event_time + duration)
                ramp = (self.time[mask] - event_time) / duration
                flux[mask] += amplitude * ramp
                flux[self.time >= event_time + duration] += amplitude

        # Add noise
        flux = self.add_noise(flux)

        metadata = {
            'tic_id': tic_id,
            'label': 'background',
            'n_events': n_events
        }

        return self.time.copy(), flux, metadata


def generate_dataset(output_dir, n_planets=500, n_non_planets=500,
                     duration_days=27.0, noise_ppm=200.0, seed=42):
    """
    Generate balanced synthetic dataset

    Args:
        output_dir: Where to save light curves
        n_planets: Number of planetary transit light curves
        n_non_planets: Number of non-planet light curves
        duration_days: Length of each light curve (TESS sector ~ 27 days)
        noise_ppm: Photon noise level
        seed: Random seed
    """

    print("="*80)
    print(" SYNTHETIC LIGHT CURVE DATASET GENERATOR")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Planets: {n_planets}")
    print(f"  Non-planets: {n_non_planets} (split: flares, EB, noise, background)")
    print(f"  Duration: {duration_days} days")
    print(f"  Noise: {noise_ppm} ppm")
    print(f"  Seed: {seed}")
    print(f"  Output: {output_dir}\n")

    # Create output directories
    output_path = Path(output_dir)
    processed_path = output_path / "processed"
    processed_path.mkdir(parents=True, exist_ok=True)

    # Initialize generator
    generator = SyntheticLightCurveGenerator(
        duration_days=duration_days,
        cadence_minutes=2.0,
        noise_ppm=noise_ppm,
        seed=seed
    )

    metadata_list = []

    # Generate planets
    print(f"\n{'='*80}")
    print(f"Generating {n_planets} planetary transits...")
    print(f"{'='*80}")

    for i in tqdm(range(n_planets), desc="Planets"):
        tic_id = 1000000 + i

        try:
            time, flux, metadata = generator.generate_planet_transit(tic_id)

            # Save as CSV
            df = pd.DataFrame({'time': time, 'flux': flux})
            csv_path = processed_path / f"{tic_id}_lightcurve.csv"
            df.to_csv(csv_path, index=False)

            metadata['curve_path'] = f"processed/{tic_id}_lightcurve.csv"
            metadata_list.append(metadata)

        except Exception as e:
            print(f"\n  Warning: Failed to generate planet {tic_id}: {e}")

    # Generate non-planets (split equally)
    n_per_type = n_non_planets // 4

    non_planet_types = [
        ('flare', generator.generate_stellar_flare, n_per_type),
        ('EB', generator.generate_eclipsing_binary, n_per_type),
        ('noise', generator.generate_pure_noise, n_per_type),
        ('background', generator.generate_background_event, n_non_planets - 3*n_per_type)  # Remainder
    ]

    offset = 2000000

    for label, generate_func, count in non_planet_types:
        print(f"\n{'='*80}")
        print(f"Generating {count} {label} light curves...")
        print(f"{'='*80}")

        for i in tqdm(range(count), desc=label.capitalize()):
            tic_id = offset + i

            try:
                time, flux, metadata = generate_func(tic_id)

                # Save as CSV
                df = pd.DataFrame({'time': time, 'flux': flux})
                csv_path = processed_path / f"{tic_id}_lightcurve.csv"
                df.to_csv(csv_path, index=False)

                metadata['curve_path'] = f"processed/{tic_id}_lightcurve.csv"
                metadata_list.append(metadata)

            except Exception as e:
                print(f"\n  Warning: Failed to generate {label} {tic_id}: {e}")

        offset += count

    # Create manifest
    print(f"\n{'='*80}")
    print("Creating manifest...")
    print(f"{'='*80}")

    manifest_df = pd.DataFrame(metadata_list)

    # Shuffle rows
    manifest_df = manifest_df.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Save manifest
    manifest_path = output_path / "manifest.csv"

    # Select key columns for manifest
    manifest_columns = ['tic_id', 'label', 'curve_path']
    if 'period' in manifest_df.columns:
        manifest_columns.append('period')
    if 'depth' in manifest_df.columns:
        manifest_columns.append('depth')

    manifest_df[manifest_columns].to_csv(manifest_path, index=False)

    print(f"\n  Saved manifest: {manifest_path}")

    # Summary statistics
    print(f"\n{'='*80}")
    print(" DATASET SUMMARY")
    print(f"{'='*80}")

    label_counts = manifest_df['label'].value_counts()
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        percentage = 100 * count / len(manifest_df)
        print(f"  {label:15s}: {count:4d} ({percentage:5.1f}%)")

    print(f"\nTotal light curves: {len(manifest_df)}")
    print(f"Points per curve: {generator.n_points}")
    print(f"Cadence: {generator.cadence_minutes} minutes")

    if 'planet' in label_counts:
        planet_df = manifest_df[manifest_df['label'] == 'planet']
        if 'period' in planet_df.columns:
            print(f"\nPlanet period range: {planet_df['period'].min():.2f} - {planet_df['period'].max():.2f} days")
        if 'depth' in planet_df.columns:
            print(f"Transit depth range: {100*planet_df['depth'].min():.3f} - {100*planet_df['depth'].max():.3f} %")

    print(f"\nDataset saved to: {output_dir}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate balanced synthetic light curve dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 1000 light curves (500 planets, 500 non-planets)
  python generate_synthetic_dataset.py --output_dir "synthetic_balanced_dataset" --n_planets 500 --n_non_planets 500

  # Custom noise level
  python generate_synthetic_dataset.py --output_dir "noisy_dataset" --noise_ppm 500 --seed 123
        """
    )

    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for synthetic dataset')
    parser.add_argument('--n_planets', type=int, default=500,
                       help='Number of planetary transit light curves (default: 500)')
    parser.add_argument('--n_non_planets', type=int, default=500,
                       help='Number of non-planet light curves (default: 500)')
    parser.add_argument('--duration_days', type=float, default=27.0,
                       help='Duration of each light curve in days (default: 27, TESS sector)')
    parser.add_argument('--noise_ppm', type=float, default=200.0,
                       help='Photon noise level in ppm (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Generate dataset
    generate_dataset(
        output_dir=args.output_dir,
        n_planets=args.n_planets,
        n_non_planets=args.n_non_planets,
        duration_days=args.duration_days,
        noise_ppm=args.noise_ppm,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
