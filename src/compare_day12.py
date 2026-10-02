import os
import csv
import numpy as np
import matplotlib.pyplot as plt


BASE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

RESULT_DIR = os.path.join(
    BASE, "results", "day12"
)

TEMPLATE_FILE = os.path.join(
    RESULT_DIR,
    "template_results.csv"
)

AI_FILE = os.path.join(
    RESULT_DIR,
    "ai_results.csv"
)


# ---------------------------------------------------------
# Read template results
# ---------------------------------------------------------

template_errors = []
template_times = []

with open(TEMPLATE_FILE, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:

        template_errors.append(
            float(row["error_pixels"])
        )

        template_times.append(
            float(row["processing_time_ms"])
        )


# ---------------------------------------------------------
# Read AI results
# ---------------------------------------------------------

ai_errors = []

with open(AI_FILE, "r") as f:

    reader = csv.DictReader(f)

    for row in reader:

        ai_errors.append(
            float(row["error_pixels"])
        )


# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

template_mean = np.mean(template_errors)
ai_mean = np.mean(ai_errors)

template_median = np.median(template_errors)
ai_median = np.median(ai_errors)

template_max = np.max(template_errors)
ai_max = np.max(ai_errors)

template_success = np.mean(
    np.array(template_errors) <= 50
) * 100

ai_success = np.mean(
    np.array(ai_errors) <= 50
) * 100

template_time = np.mean(template_times)


# ---------------------------------------------------------
# Print comparison
# ---------------------------------------------------------

print("=" * 70)
print("DAY 12 - TEMPLATE MATCHING VS SPATIAL AI")
print("=" * 70)

print()
print("Metric                    Template       AI")

print(
    f"Mean error               "
    f"{template_mean:8.2f} px   "
    f"{ai_mean:8.2f} px"
)

print(
    f"Median error             "
    f"{template_median:8.2f} px   "
    f"{ai_median:8.2f} px"
)

print(
    f"Maximum error            "
    f"{template_max:8.2f} px   "
    f"{ai_max:8.2f} px"
)

print(
    f"Success rate             "
    f"{template_success:8.2f}%   "
    f"{ai_success:8.2f}%"
)

print(
    f"Template processing time "
    f"{template_time:8.2f} ms"
)

print("=" * 70)


# ---------------------------------------------------------
# Save comparison CSV
# ---------------------------------------------------------

comparison_file = os.path.join(
    RESULT_DIR,
    "comparison_summary.csv"
)

with open(
    comparison_file,
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "metric",
        "template_matching",
        "spatial_ai"
    ])

    writer.writerow([
        "mean_error_pixels",
        template_mean,
        ai_mean
    ])

    writer.writerow([
        "median_error_pixels",
        template_median,
        ai_median
    ])

    writer.writerow([
        "maximum_error_pixels",
        template_max,
        ai_max
    ])

    writer.writerow([
        "success_rate_percent",
        template_success,
        ai_success
    ])

    writer.writerow([
        "mean_template_processing_time_ms",
        template_time,
        "N/A"
    ])


# ---------------------------------------------------------
# Error comparison graph
# ---------------------------------------------------------

methods = [
    "Template Matching",
    "Spatial AI"
]

mean_errors = [
    template_mean,
    ai_mean
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    mean_errors
)

plt.ylabel("Mean Localization Error (pixels)")
plt.title(
    "Template Matching vs Spatial AI"
)

plt.tight_layout()

error_graph = os.path.join(
    RESULT_DIR,
    "mean_error_comparison.png"
)

plt.savefig(
    error_graph,
    dpi=150
)

plt.close()


# ---------------------------------------------------------
# Success rate graph
# ---------------------------------------------------------

success_rates = [
    template_success,
    ai_success
]

plt.figure(figsize=(8, 6))

plt.bar(
    methods,
    success_rates
)

plt.ylabel("Success Rate (%)")
plt.title(
    "Localization Success Rate"
)

plt.ylim(0, 105)

plt.tight_layout()

success_graph = os.path.join(
    RESULT_DIR,
    "success_rate_comparison.png"
)

plt.savefig(
    success_graph,
    dpi=150
)

plt.close()


print()
print("Comparison CSV:")
print(comparison_file)

print()
print("Graphs generated:")
print(error_graph)
print(success_graph)

print()
print("DAY 12 COMPARISON COMPLETE")
