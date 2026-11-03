# Preliminary leakage audit

## Prediction and analysis points

The source paper describes booking/change-log snapshots relative to the day before arrival, not the moment a booking was created [1, sections 2-3]. We therefore estimate recorded ADR and late cancellation/no-show risk at the **end of the day before scheduled arrival**. The supervised risk set requires `LeadTime >= 1` and `ReservationStatusDate > ArrivalDate - 1 day`. A terminal event on the snapshot day has no time-of-day information and is conservatively excluded. Earlier cancellations are outside this task's population, rather than examples of future cancellation.

This reconstructs active-at-snapshot membership from terminal dates. It assumes terminal status was not reversed and that retained booking/change-log attributes represent the documented snapshot. The CSV cannot prove every field's timestamp. Performance must not be presented as booking-creation cancellation forecasting or as performance on all 79,330 bookings.

Clustering is descriptive full-H2 segmentation of recorded booking profiles. It uses no outcome columns, eligibility outcome filters, supervised models, or supervised score to select groups. Its transformations and labels never enter the supervised tasks.

## All 31 source columns

R/C means both supervised tasks. A retained source is a substantive assumption based on its meaning and documented booking/change-log provenance; it is not a guarantee about an unobserved event history.

| Source column | Regression / classification | Clustering | Rationale |
| --- | --- | --- | --- |
| IsCanceled | Excluded / target | Excluded | Final outcome; never an input to any task. |
| LeadTime | Retained numeric | log1p | Booking lead time is known at the snapshot; eligibility requires at least one day. |
| ArrivalDateYear | Retained numeric | Excluded | Scheduled year is known; it also enables chronological splitting. |
| ArrivalDateMonth | Sine/cosine derived features | Excluded | Known scheduled month; cyclic encoding avoids an artificial December/January gap. |
| ArrivalDateWeekNumber | Excluded | Excluded | Redundant calendar information; no need for another correlated calendar representation. |
| ArrivalDateDayOfMonth | Retained numeric | Excluded | Scheduled day at the documented snapshot; also used to reconstruct arrival date. |
| StaysInWeekendNights | Retained plus derived features | Via total nights/weekend share | Booked length at snapshot, not an observed future-stay correction. Snapshot provenance is an assumption. |
| StaysInWeekNights | Retained plus derived features | Via total nights | Same booked-stay interpretation and limitation. |
| Adults | Retained numeric | Retained numeric | Booking party composition at snapshot. |
| Children | Retained numeric; train median imputation | Median imputation | Booking party composition; four source unknowns. |
| Babies | Retained numeric | Retained numeric | Booking party composition; rare values can dominate scaled cluster distances. |
| Meal | Retained categorical | Excluded | Meal booked at snapshot; keep source categories including no-meal codes. |
| Country | Retained categorical; UNKNOWN imputation | Excluded | Available booking-origin code; 24 unknowns. Country is descriptive, not a causal explanation. |
| MarketSegment | Retained categorical | Retained categorical | Recorded booking segment, not an outcome. |
| DistributionChannel | Retained categorical | Excluded | Recorded acquisition channel. |
| IsRepeatedGuest | Retained numeric binary | Excluded | Paper derives it using a customer profile created before the current booking. |
| PreviousCancellations | Excluded | Excluded | Previous-booking outcome timing cannot be audited at the chosen snapshot from this CSV. Conservative removal. |
| PreviousBookingsNotCanceled | Excluded | Excluded | Same unresolved as-of outcome-history risk. |
| ReservedRoomType | Retained categorical | Retained categorical | Room type booked at snapshot, rather than eventual assignment. |
| AssignedRoomType | Excluded | Excluded | Conservative exclusion of an operational assignment that may change near check-in. |
| BookingChanges | Excluded | Excluded | Source describes a count accumulated through check-in/cancellation, potentially after the prediction point. |
| DepositType | Excluded | Excluded | Paper derives it from transactions before arrival/cancellation; this can include post-snapshot payments. |
| Agent | Retained categorical ID | Excluded | Booking agency known at snapshot; NULL means not applicable. Never treat ID magnitude as numeric. |
| Company | Retained categorical ID | Excluded | Booking/paying entity at snapshot; NULL means not applicable. Sparse categories can limit generalization. |
| DaysInWaitingList | Excluded | Excluded | Confirmation-related duration may be known for some active bookings, but timing/completeness is not established for all rows. |
| CustomerType | Retained categorical | Retained categorical | Booking relationship type at snapshot. |
| ADR | Target / excluded | Excluded | Final recorded rate is the regression target and may include later transaction information. |
| RequiredCarParkingSpaces | Excluded | Excluded | Requests may change; omitted conservatively because request timestamps are unavailable. |
| TotalOfSpecialRequests | Excluded | Excluded | Aggregate request count may include later additions. |
| ReservationStatus | Excluded as predictor | Excluded | Direct final-outcome information; semantic validation only. |
| ReservationStatusDate | Eligibility and label-availability checks only | Excluded | Used solely to reconstruct active-at-snapshot membership and purge unavailable labels; never encoded as input. |

## Fit/evaluation isolation

- Training arrival dates: 1 July 2015 through 31 December 2016; retain only labels resolved before 31 December 2016 (the first validation snapshot date).
- Validation arrivals: 1 January through 30 April 2017; retain only labels resolved before 30 April 2017 (the first test snapshot date).
- Test arrivals: 1 May through 31 August 2017. No supervised estimator receives test inputs or target values for prediction/scoring in Milestone 1.
- All imputers, encoders, and scalers in a supervised pipeline are fitted on training rows only. Unseen validation categories produce an all-zero one-hot block, without refitting.
- Derived calendar and stay features are deterministic and contain no learned global statistics. Numeric medians and scaling remain inside the fitted pipeline [2].
- Full-source EDA is restricted to schema, missingness, duplicates, date coverage, and structural domain flags. Target distributions, target associations, and numeric outlier summaries use training data. Validation is used only for reported preliminary evaluation.
- Exact source duplicates are retained because the CSV has no booking identifiers and group bookings can share values. Identical predictor groups have no overlap across chronological partitions; this is explicitly checked. Hidden customer/agency dependence remains possible.
- No tuning or model-driven feature revision is performed against the final test set. Models are preliminary and untuned; default-threshold weaknesses are reported without optimizing the threshold in this milestone.
- Independent clustering fits its own preprocessing on all H2 booking inputs. No outcome values select variables, k, or parameters. k=3 is a fixed exploratory baseline, not evidence that three is optimal. PCA is used only to visualize those clusters; it is not a completed supervised PCA/LDA comparison.

## Remaining risks and next checks

The retrospective active-booking filter narrows the population and can be affected by date granularity, terminal-status reversals, and source snapshot inconsistencies. Boundary purging removes long-running outcomes and may introduce a small selection effect. Chronological cohorts differ seasonally; arrival-cohort performance is not a full rolling historical deployment simulation. The data has neither unique booking IDs nor timestamped feature histories, so independence and as-of correctness cannot be fully certified. Validate group/customer generalization, feature availability, and temporal-fold label purging before Milestone 2 tuning. Full-source clustering is descriptive and must never be reused as a supervised feature learned from future rows.

## References

[1] Nuno Antonio, Ana de Almeida, and Luis Nunes. Hotel booking demand datasets. Data in Brief 22 (2019), 41-49. https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ (sections 2-3 and Table 1; also available as https://www.ebi.ac.uk/europepmc/webservices/rest/PMC6297060/fullTextXML).

[2] scikit-learn. Pipelines and composite estimators. https://scikit-learn.org/stable/modules/compose.html
