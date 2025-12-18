"""
Inspect Kepler KSOC-5104 data and explain lightkurve usage.

lightkurve downloads ACTUAL observed light curve data from MAST.
It does NOT provide simulated data - only real observations.
"""
import os
import sys

print("=" * 60)
print("UNDERSTANDING LIGHTKURVE AND KEPLER DATA")
print("=" * 60)

print("""
WHAT IS LIGHTKURVE?
-------------------
lightkurve is a Python package that downloads ACTUAL OBSERVED
light curve data from NASA's MAST archive. It provides:

1. Real Kepler observations (2009-2013)
2. Real K2 observations (2014-2018)
3. Real TESS observations (2018-present)

It does NOT:
- Generate synthetic/simulated data
- Provide transit injection test results
- Create artificial transits

HOW TO USE IT:
--------------
import lightkurve as lk

# Download real Kepler data for a star
lc = lk.search_lightcurve("KIC 10666592", mission="Kepler").download()

# Download real TESS data
lc = lk.search_lightcurve("TIC 307210830", mission="TESS").download()
""")

print("\n" + "=" * 60)
print("INSPECTING YOUR KSOC-5104 DATA")
print("=" * 60)

kepler_dir = r"E:\kepler_data\KSOC-5104"

if os.path.exists(kepler_dir):
    files = os.listdir(kepler_dir)
    print(f"\nFound {len(files)} files in {kepler_dir}")

    # Show sample files
    fits_files = [f for f in files if f.endswith('.fits')]
    print(f"FITS files: {len(fits_files)}")

    if fits_files:
        print("\nSample filenames:")
        for f in fits_files[:5]:
            print(f"  {f}")

        # Try to read one
        try:
            from astropy.io import fits

            sample_path = os.path.join(kepler_dir, fits_files[0])
            print(f"\n[Reading] {fits_files[0]}")

            with fits.open(sample_path) as hdul:
                print(f"  HDU list: {[h.name for h in hdul]}")

                for i, hdu in enumerate(hdul):
                    print(f"\n  HDU[{i}] = {hdu.name}")
                    if hasattr(hdu, 'columns') and hdu.columns:
                        print(f"    Columns: {[c.name for c in hdu.columns]}")
                        # Show sample data
                        if len(hdu.data) > 0:
                            print(f"    Rows: {len(hdu.data)}")
                            print(f"    First row sample:")
                            for col in list(hdu.columns.names)[:6]:
                                try:
                                    val = hdu.data[col][0]
                                    print(f"      {col}: {val}")
                                except:
                                    pass
        except Exception as e:
            print(f"  Error reading FITS: {e}")
else:
    print(f"\nDirectory not found: {kepler_dir}")
    print("Make sure the drive is mounted and path is correct.")

print("\n" + "=" * 60)
print("HOW TO GET USABLE TRAINING DATA")
print("=" * 60)

print("""
The KSOC-5104 files contain INJECTION TEST RESULTS - metadata about
synthetic transits that were injected into Kepler data for testing.
They do NOT contain the actual flux time series.

OPTIONS TO GET DIVERSE TRAINING DATA:
-------------------------------------

1. USE LIGHTKURVE TO DOWNLOAD ACTUAL KEPLER LIGHT CURVES:

   import lightkurve as lk

   # Get confirmed exoplanet host stars
   confirmed_planets = ["Kepler-10", "Kepler-22", "Kepler-62", ...]

   for name in confirmed_planets:
       lc = lk.search_lightcurve(name, mission="Kepler").download_all()
       # Process and save as training windows

   This gives you REAL data with KNOWN planets.

2. DOWNLOAD KEPLER CONFIRMED PLANET CATALOG:
   - NASA Exoplanet Archive has all confirmed Kepler planets
   - We can get KIC IDs and download their light curves
   - ~2,700 confirmed Kepler planets available

3. COMBINE WITH EXISTING TESS DATA:
   - Keep your Sector 1 training data
   - Add Kepler confirmed planet light curves
   - Train on combined TESS + Kepler dataset

RECOMMENDED APPROACH:
--------------------
Create a script that:
1. Gets list of ~100-500 confirmed Kepler planets from NASA archive
2. Downloads their light curves using lightkurve
3. Processes them into training windows (same as Sector 1)
4. Combines with Sector 1 data for training
""")

print("\n" + "=" * 60)
print("READY TO PROCEED?")
print("=" * 60)
print("""
I can create a script to download Kepler confirmed exoplanet
light curves using lightkurve. This will give you:

- Real Kepler flux data (not just metadata)
- Known positive examples (confirmed planets)
- Different instrument/noise characteristics than TESS
- Better generalization when combined with TESS data

The download will take some time (~1-2 hours for 500 planets)
but gives you properly labeled training data.
""")
