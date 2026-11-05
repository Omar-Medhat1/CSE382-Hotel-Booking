# Clustering Verification and Segment Interpretation

**Member:** Ahmed Mokhtar  
**Student ID:** 23P0081  
**Project:** CSE382 — Hotel Booking Analysis  
**Date:** 5 November 2026

## 1. Clustering Baseline

The existing K-Means baseline was reproduced using the City Hotel dataset (H2.csv), containing 79,330 bookings.

The clustering workflow uses three clusters, 26 transformed features, and a fixed random seed of 382.

The reproduced results were:

| Metric | Value |
|---|---:|
| Silhouette score (2,000 sampled rows) | 0.219668 |
| Davies-Bouldin index | 1.738646 |
| Inertia | 445210.0458 |
| Number of clusters | 3 |

The clustering results are exploratory and have not been optimized.

## 2. Verification

An additional clustering integrity check was implemented locally in `src/verification.py`.

The check verifies:

- Complete coverage of all 79,330 source rows.
- Unique source-row assignments and matching row order.
- Identical predictions after reloading the saved clustering pipeline.
- Agreement between assignment counts and exported cluster profiles.

The updated `main.py verify` command completed successfully and reported `clustering_reload_and_coverage: passed`.

## 3. Segment Interpretation

**Cluster 0 — Short-notice, smaller-party bookings**

Contains 22,063 bookings (27.8%). Median lead time is 13 days, median stay is 2 nights, and mean adults per booking is approximately 1.39. This cluster is characterized by relatively short booking lead times and smaller parties.

**Cluster 1 — Early-booking, mostly adult parties**

Contains 52,179 bookings (65.8%). Median lead time is 109 days, median stay is 3 nights, and mean adults per booking is approximately 2.04. The mean number of children is zero. This group represents earlier bookings with predominantly adult parties.

**Cluster 2 — Bookings with children**

Contains 5,088 bookings (6.4%). Median lead time is 64 days, median stay is 3 nights, and mean children per booking is approximately 1.42. This cluster is distinguished by its higher number of children.

Across all three clusters, the most common market segment is Online TA, the most common customer type is Transient, and the most common reserved room type is A.

## 4. PCA Visualization

PCA was used to visualize the 26-dimensional clustering representation in two dimensions. The first two principal components explain approximately 34.52% of the total variance.

The plot displays an 8,000-booking sample, while the cluster-size chart represents all 79,330 bookings.

The visualization shows some separation between clusters, but also overlapping regions. Since approximately 65.48% of the variance is not represented in the two-dimensional projection, the plot alone cannot establish separation in the original feature space.

## 5. Limitations and Future Work

The silhouette score of approximately 0.22 indicates limited cluster separation. The current choice of three clusters is an exploratory baseline rather than an optimized result.

Potential Milestone 2 work includes comparing other cluster counts, investigating feature scaling and weighting, testing hierarchical clustering and DBSCAN, and evaluating cluster stability across repeated runs.

Clustering is kept separate from supervised regression and classification. Cancellation outcomes and ADR are not used to define the clusters or choose clustering parameters.

## 6. Verification Record

- Dataset structural validation: passed.
- Clustering baseline reproduction: matched recorded metrics.
- Existing integrated verification: passed.
- Clustering reload and assignment verification: passed.
- Original eight automated tests: passed before the clustering verification modification.
- Automated tests after modification: 8 tests passed (5 November 2026).
- GitHub pull request and teammate review: pending.

The modification is currently local and requires coordination with the member responsible for integrated verification before merging.
## Integration status on 5 November 2026

PR #6 is merged. Final integration retained the clustering coverage, row-order,
reload and profile-count checks alongside classification consistency checks and
Hala's explicit runtime validation. Normal and Python -O verification passed.
The pending statements above describe the original member handoff, not the
current integration status.
