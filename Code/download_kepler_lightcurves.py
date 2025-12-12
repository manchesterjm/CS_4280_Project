"""
Download Kepler light curves for confirmed exoplanets using lightkurve.

This script downloads light curves from MAST for fine-tuning the TESS model
on Kepler data (target: 5% Kepler mix = ~1400 windows = ~470 light curves).

Usage:
    python download_kepler_lightcurves.py --output_dir "D:/kepler_downloads" --n_targets 500
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def get_confirmed_kepler_planets():
    """
    Return a list of confirmed Kepler planet host stars.

    These are well-known Kepler systems with confirmed planets,
    suitable for training/fine-tuning exoplanet detection models.
    """
    # Confirmed Kepler planet host stars (KOI = Kepler Object of Interest)
    # Each of these has at least one confirmed planet
    kepler_targets = [
        # Famous multi-planet systems
        "Kepler-11", "Kepler-20", "Kepler-22", "Kepler-36", "Kepler-37",
        "Kepler-42", "Kepler-62", "Kepler-69", "Kepler-78", "Kepler-90",
        "Kepler-186", "Kepler-438", "Kepler-442", "Kepler-452", "Kepler-1647",

        # Additional confirmed systems (Kepler-1 through Kepler-100)
        "Kepler-1", "Kepler-2", "Kepler-3", "Kepler-4", "Kepler-5",
        "Kepler-6", "Kepler-7", "Kepler-8", "Kepler-9", "Kepler-10",
        "Kepler-12", "Kepler-13", "Kepler-14", "Kepler-15", "Kepler-16",
        "Kepler-17", "Kepler-18", "Kepler-19", "Kepler-21", "Kepler-23",
        "Kepler-24", "Kepler-25", "Kepler-26", "Kepler-27", "Kepler-28",
        "Kepler-29", "Kepler-30", "Kepler-31", "Kepler-32", "Kepler-33",
        "Kepler-34", "Kepler-35", "Kepler-38", "Kepler-39", "Kepler-40",
        "Kepler-41", "Kepler-43", "Kepler-44", "Kepler-45", "Kepler-46",
        "Kepler-47", "Kepler-48", "Kepler-49", "Kepler-50", "Kepler-51",
        "Kepler-52", "Kepler-53", "Kepler-54", "Kepler-55", "Kepler-56",
        "Kepler-57", "Kepler-58", "Kepler-59", "Kepler-60", "Kepler-61",
        "Kepler-63", "Kepler-64", "Kepler-65", "Kepler-66", "Kepler-67",
        "Kepler-68", "Kepler-70", "Kepler-71", "Kepler-72", "Kepler-73",
        "Kepler-74", "Kepler-75", "Kepler-76", "Kepler-77", "Kepler-79",
        "Kepler-80", "Kepler-81", "Kepler-82", "Kepler-83", "Kepler-84",
        "Kepler-85", "Kepler-86", "Kepler-87", "Kepler-88", "Kepler-89",
        "Kepler-91", "Kepler-92", "Kepler-93", "Kepler-94", "Kepler-95",
        "Kepler-96", "Kepler-97", "Kepler-98", "Kepler-99", "Kepler-100",

        # More confirmed systems (Kepler-101 through Kepler-200)
        "Kepler-101", "Kepler-102", "Kepler-103", "Kepler-104", "Kepler-105",
        "Kepler-106", "Kepler-107", "Kepler-108", "Kepler-109", "Kepler-110",
        "Kepler-111", "Kepler-112", "Kepler-113", "Kepler-114", "Kepler-115",
        "Kepler-116", "Kepler-117", "Kepler-118", "Kepler-119", "Kepler-120",
        "Kepler-121", "Kepler-122", "Kepler-123", "Kepler-124", "Kepler-125",
        "Kepler-126", "Kepler-127", "Kepler-128", "Kepler-129", "Kepler-130",
        "Kepler-131", "Kepler-132", "Kepler-133", "Kepler-134", "Kepler-135",
        "Kepler-136", "Kepler-137", "Kepler-138", "Kepler-139", "Kepler-140",
        "Kepler-141", "Kepler-142", "Kepler-143", "Kepler-144", "Kepler-145",
        "Kepler-146", "Kepler-147", "Kepler-148", "Kepler-149", "Kepler-150",
        "Kepler-151", "Kepler-152", "Kepler-153", "Kepler-154", "Kepler-155",
        "Kepler-156", "Kepler-157", "Kepler-158", "Kepler-159", "Kepler-160",
        "Kepler-161", "Kepler-162", "Kepler-163", "Kepler-164", "Kepler-165",
        "Kepler-166", "Kepler-167", "Kepler-168", "Kepler-169", "Kepler-170",
        "Kepler-171", "Kepler-172", "Kepler-173", "Kepler-174", "Kepler-175",
        "Kepler-176", "Kepler-177", "Kepler-178", "Kepler-179", "Kepler-180",
        "Kepler-181", "Kepler-182", "Kepler-183", "Kepler-184", "Kepler-185",
        "Kepler-187", "Kepler-188", "Kepler-189", "Kepler-190", "Kepler-191",
        "Kepler-192", "Kepler-193", "Kepler-194", "Kepler-195", "Kepler-196",
        "Kepler-197", "Kepler-198", "Kepler-199", "Kepler-200",

        # Additional systems (Kepler-201 through Kepler-300)
        "Kepler-201", "Kepler-202", "Kepler-203", "Kepler-204", "Kepler-205",
        "Kepler-206", "Kepler-207", "Kepler-208", "Kepler-209", "Kepler-210",
        "Kepler-211", "Kepler-212", "Kepler-213", "Kepler-214", "Kepler-215",
        "Kepler-216", "Kepler-217", "Kepler-218", "Kepler-219", "Kepler-220",
        "Kepler-221", "Kepler-222", "Kepler-223", "Kepler-224", "Kepler-225",
        "Kepler-226", "Kepler-227", "Kepler-228", "Kepler-229", "Kepler-230",
        "Kepler-231", "Kepler-232", "Kepler-233", "Kepler-234", "Kepler-235",
        "Kepler-236", "Kepler-237", "Kepler-238", "Kepler-239", "Kepler-240",
        "Kepler-241", "Kepler-242", "Kepler-243", "Kepler-244", "Kepler-245",
        "Kepler-246", "Kepler-247", "Kepler-248", "Kepler-249", "Kepler-250",
        "Kepler-251", "Kepler-252", "Kepler-253", "Kepler-254", "Kepler-255",
        "Kepler-256", "Kepler-257", "Kepler-258", "Kepler-259", "Kepler-260",
        "Kepler-261", "Kepler-262", "Kepler-263", "Kepler-264", "Kepler-265",
        "Kepler-266", "Kepler-267", "Kepler-268", "Kepler-269", "Kepler-270",
        "Kepler-271", "Kepler-272", "Kepler-273", "Kepler-274", "Kepler-275",
        "Kepler-276", "Kepler-277", "Kepler-278", "Kepler-279", "Kepler-280",
        "Kepler-281", "Kepler-282", "Kepler-283", "Kepler-284", "Kepler-285",
        "Kepler-286", "Kepler-287", "Kepler-288", "Kepler-289", "Kepler-290",
        "Kepler-291", "Kepler-292", "Kepler-293", "Kepler-294", "Kepler-295",
        "Kepler-296", "Kepler-297", "Kepler-298", "Kepler-299", "Kepler-300",

        # More systems (Kepler-301 through Kepler-450)
        "Kepler-301", "Kepler-302", "Kepler-303", "Kepler-304", "Kepler-305",
        "Kepler-306", "Kepler-307", "Kepler-308", "Kepler-309", "Kepler-310",
        "Kepler-311", "Kepler-312", "Kepler-313", "Kepler-314", "Kepler-315",
        "Kepler-316", "Kepler-317", "Kepler-318", "Kepler-319", "Kepler-320",
        "Kepler-321", "Kepler-322", "Kepler-323", "Kepler-324", "Kepler-325",
        "Kepler-326", "Kepler-327", "Kepler-328", "Kepler-329", "Kepler-330",
        "Kepler-331", "Kepler-332", "Kepler-333", "Kepler-334", "Kepler-335",
        "Kepler-336", "Kepler-337", "Kepler-338", "Kepler-339", "Kepler-340",
        "Kepler-341", "Kepler-342", "Kepler-343", "Kepler-344", "Kepler-345",
        "Kepler-346", "Kepler-347", "Kepler-348", "Kepler-349", "Kepler-350",
        "Kepler-351", "Kepler-352", "Kepler-353", "Kepler-354", "Kepler-355",
        "Kepler-356", "Kepler-357", "Kepler-358", "Kepler-359", "Kepler-360",
        "Kepler-361", "Kepler-362", "Kepler-363", "Kepler-364", "Kepler-365",
        "Kepler-366", "Kepler-367", "Kepler-368", "Kepler-369", "Kepler-370",
        "Kepler-371", "Kepler-372", "Kepler-373", "Kepler-374", "Kepler-375",
        "Kepler-376", "Kepler-377", "Kepler-378", "Kepler-379", "Kepler-380",
        "Kepler-381", "Kepler-382", "Kepler-383", "Kepler-384", "Kepler-385",
        "Kepler-386", "Kepler-387", "Kepler-388", "Kepler-389", "Kepler-390",
        "Kepler-391", "Kepler-392", "Kepler-393", "Kepler-394", "Kepler-395",
        "Kepler-396", "Kepler-397", "Kepler-398", "Kepler-399", "Kepler-400",
        "Kepler-401", "Kepler-402", "Kepler-403", "Kepler-404", "Kepler-405",
        "Kepler-406", "Kepler-407", "Kepler-408", "Kepler-409", "Kepler-410",
        "Kepler-411", "Kepler-412", "Kepler-413", "Kepler-414", "Kepler-415",
        "Kepler-416", "Kepler-417", "Kepler-418", "Kepler-419", "Kepler-420",
        "Kepler-421", "Kepler-422", "Kepler-423", "Kepler-424", "Kepler-425",
        "Kepler-426", "Kepler-427", "Kepler-428", "Kepler-429", "Kepler-430",
        "Kepler-431", "Kepler-432", "Kepler-433", "Kepler-434", "Kepler-435",
        "Kepler-436", "Kepler-437", "Kepler-439", "Kepler-440", "Kepler-441",
        "Kepler-443", "Kepler-444", "Kepler-445", "Kepler-446", "Kepler-447",
        "Kepler-448", "Kepler-449", "Kepler-450",
    ]
    return kepler_targets


def download_kepler_data(output_dir, n_targets=500, skip_existing=True):
    """
    Download Kepler light curves using lightkurve.

    Args:
        output_dir: Directory to save downloaded light curves
        n_targets: Number of targets to download
        skip_existing: Skip targets that already have data
    """
    try:
        import lightkurve as lk
    except ImportError:
        print("ERROR: lightkurve not installed.")
        print("Install with: pip install lightkurve")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets = get_confirmed_kepler_planets()[:n_targets]
    print(f"Attempting to download {len(targets)} Kepler targets")
    print(f"Output directory: {output_dir}")

    successful = 0
    failed = 0
    skipped = 0
    download_log = []

    for i, target in enumerate(targets):
        target_file = output_dir / f"{target.replace(' ', '_')}_lightcurve.csv"

        if skip_existing and target_file.exists():
            print(f"[{i+1}/{len(targets)}] SKIP: {target} (already exists)")
            skipped += 1
            continue

        print(f"[{i+1}/{len(targets)}] Downloading {target}...", end=" ")

        try:
            # Search for Kepler light curves
            search_result = lk.search_lightcurve(
                target,
                mission="Kepler",
                author="Kepler"
            )

            if len(search_result) == 0:
                print("NO DATA")
                failed += 1
                download_log.append({
                    'target': target,
                    'status': 'no_data',
                    'n_quarters': 0
                })
                continue

            # Download and stitch all quarters
            lc_collection = search_result.download_all()
            stitched_lc = lc_collection.stitch()

            # Extract time and flux
            time = stitched_lc.time.value
            flux = stitched_lc.flux.value

            # Remove NaN values
            valid_mask = ~np.isnan(flux) & ~np.isnan(time)
            time = time[valid_mask]
            flux = flux[valid_mask]

            if len(flux) < 1000:
                print(f"TOO SHORT ({len(flux)} points)")
                failed += 1
                download_log.append({
                    'target': target,
                    'status': 'too_short',
                    'n_quarters': len(search_result),
                    'n_points': len(flux)
                })
                continue

            # Save to CSV
            df = pd.DataFrame({'time': time, 'flux': flux})
            df.to_csv(target_file, index=False)

            print(f"OK ({len(flux):,} points, {len(search_result)} quarters)")
            successful += 1
            download_log.append({
                'target': target,
                'status': 'success',
                'n_quarters': len(search_result),
                'n_points': len(flux),
                'file': str(target_file)
            })

        except (OSError, ValueError, RuntimeError) as exc:
            print(f"ERROR: {exc}")
            failed += 1
            download_log.append({
                'target': target,
                'status': 'error',
                'error': str(exc)
            })

    # Save download log
    log_df = pd.DataFrame(download_log)
    log_df.to_csv(output_dir / 'download_log.csv', index=False)

    print(f"\n{'='*60}")
    print("DOWNLOAD SUMMARY")
    print(f"{'='*60}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {successful + failed + skipped}")
    print(f"\nFiles saved to: {output_dir}")
    print(f"Log saved to: {output_dir / 'download_log.csv'}")

    # Calculate expected windows
    expected_windows = successful * 3  # 3 windows per light curve
    print(f"\nExpected training windows: {expected_windows}")
    print(f"Target for 5% mix: ~1,400 windows (~467 light curves)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Download Kepler light curves for fine-tuning'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=r'D:\CS_4280_Project\kepler_downloads',
        help='Directory to save downloaded light curves'
    )
    parser.add_argument(
        '--n_targets',
        type=int,
        default=500,
        help='Number of targets to download (default: 500)'
    )
    parser.add_argument(
        '--skip_existing',
        action='store_true',
        default=True,
        help='Skip targets that already have data'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    download_kepler_data(
        output_dir=args.output_dir,
        n_targets=args.n_targets,
        skip_existing=args.skip_existing
    )


if __name__ == '__main__':
    main()
