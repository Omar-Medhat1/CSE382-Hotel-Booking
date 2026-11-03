"""Build a paginated PDF and matching Markdown from recorded experiment results."""
import html
import json
import re
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak, KeepTogether)

from src.data.loading import ROOT

TEAL = colors.HexColor("#186B79")
NAVY = colors.HexColor("#183243")
LIGHT = colors.HexColor("#EFF5F6")
MUTED = colors.HexColor("#50616C")
WIDTH = A4[0] - 84


def build():
    run = json.loads((ROOT / "results/run_summary.json").read_text(encoding="utf-8"))
    team = json.loads((ROOT / "docs/team.json").read_text(encoding="utf-8"))
    if team["team_size"] != len(team["members"]):
        raise ValueError("Team size does not match the roster.")
    eda, models, cluster = run["eda"], run["models"], run["clustering"]
    ridge, mean = models["ridge"], models["training_mean"]
    logit, prior = models["logistic_regression"], models["training_prior"]
    counts = run["split"]["partition_counts"]
    profiles = pd.read_csv(ROOT / "results/tables/cluster_profiles.csv")
    sections = []

    def page(title, subtitle, blocks):
        sections.append((title, subtitle, blocks))

    page("Milestone 1", "CITY HOTEL BOOKING DEMAND | CSE382 | 3 NOVEMBER 2026", [
        ("p", "<b>Data preparation and baseline experiments</b>"),
        ("p", "Technical progress report for H2 (City Hotel). This version contains executed experiments and reproducible source code. Responsibilities are assigned across the six confirmed members below. Individual completion evidence and other members' GitHub accounts remain pending before submission."),
        ("h", "Project team and assigned responsibilities"),
        ("table", ["Student name", "Student ID", "Assigned responsibility"],
         [[member["name"], member["student_id"], member["responsibility"]] for member in team["members"]],
         [170, 85, WIDTH - 255]),
        ("table", ["Area", "Measured result / status"], [
            ["Source verified", "79,330 records; 31 original columns"],
            ["ADR regression", f"Ridge validation MAE {ridge['mae']:.2f}; RMSE {ridge['rmse']:.2f}; R-squared {ridge['r2']:.3f}"],
            ["Cancellation ranking", f"Logistic validation ROC-AUC {logit['roc_auc']:.3f}; average precision {logit['average_precision']:.4f}"],
            ["Cancellation decisions", f"Default-threshold recall {logit['recall']:.2%}; a material baseline weakness"],
            ["Unsupervised baseline", f"K-Means, k=3; silhouette {cluster['silhouette']:.3f} (2,000-row sample)"],
            ["Final supervised test", f"{counts['test']:,} held-out records; no test predictions or scores produced"],
        ], [145, WIDTH-145]),
        ("h", "What this milestone completes"),
        ("p", "Modular Python implementation, source verification, training EDA, reproducible temporal partitions, reusable preprocessing, one untuned model per required task, preliminary metrics, a dedicated leakage audit, and a plan for the remaining project."),
        ("h", "Scope that governs every result"),
        ("p", "Supervised predictions apply to bookings still active at the end of the day before scheduled arrival. They estimate recorded ADR and late cancellation/no-show risk. Results do not describe cancellation prediction when a booking is first created, or performance on the entire source population."),
        ("p", "Repository: <link href='https://github.com/Omar-Medhat1/CSE382-Hotel-Booking' color='#186B79'>Omar-Medhat1 / CSE382-Hotel-Booking</link>. Milestone 1 deadline in the supplied brief: 5 November 2026."),
    ])

    page("01 / Source and data quality", "SOURCE-WIDE STRUCTURE; TRAINING-ONLY TARGET EXPLORATION", [
        ("p", "H2.csv was downloaded from the mirror specified in the course brief. Dimensions, all original source labels, field widths, the cancellation domain, status-label agreement, dates, nonnegative ADR, and a SHA-256 fingerprint were checked. H1 was not loaded or merged."),
        ("table", ["Check", "Observed value"], [
            ["Arrival coverage", f"{eda['arrival_first']} through {eda['arrival_last']}"],
            ["Unknown Children / Country", f"{eda['unknown_counts']['Children']} / {eda['unknown_counts']['Country']}"],
            ["Agent / Company not applicable", f"{eda['not_applicable_ids']['Agent']:,} / {eda['not_applicable_ids']['Company']:,}"],
            ["Duplicate rows after first occurrence", f"{eda['exact_duplicate_rows_after_first']:,}"],
            ["Zero nights / zero total guests", f"{eda['structural_flags']['zero_night_bookings']} / {eda['structural_flags']['zero_guest_bookings']}"],
            ["Zero adults / negative count entries", f"{eda['structural_flags']['adults_zero']} / {eda['structural_flags']['negative_count_values']}"],
        ], [235, WIDTH-235]),
        ("p", "Strip padded text. Convert genuine unknown markers in Children and Country to missing values. The paper explicitly defines NULL in Agent/Company as <b>not applicable</b>, so preserve it as a categorical level; do not median-impute these IDs [1]. Source names remain unchanged."),
        ("image", "missingness_and_training_target.png", WIDTH, 166),
        ("p", "Retain duplicates: identical feature rows may represent real group bookings, and no unique booking ID is supplied. Keep same-date predictor groups within chronological cohorts and verify zero cross-partition overlap. Flag unusual values without inventing corrections or deleting valid rare bookings. Structural flags are not automatic removal rules."),
        ("p", "The full fingerprint is recorded in config.json and results/run_summary.json. Source schema, category cardinalities, and training numeric ranges are exported as CSV tables."),
    ])

    page("02 / Prediction point and partitions", "AN ACTIVE-BOOKING RISK SET WITH CHRONOLOGICAL HOLDOUTS", [
        ("p", "The source paper explains that many booking/change-log values were extracted relative to the day before arrival [1, sections 2-3]. A booking-creation forecast would therefore rely on unverified earlier values. The chosen snapshot is the <b>end of the day before scheduled arrival</b>."),
        ("p", "Require LeadTime >= 1 and final ReservationStatusDate strictly after ArrivalDate minus one day. This excludes bookings created on arrival day, known earlier terminal outcomes, and same-snapshot-day outcomes whose event order cannot be determined. Terminal dates define eligibility only; they are never predictors."),
        ("table", ["Partition", "Arrival cohort / label availability", "Records"], [
            ["Training", "Before 1 Jan 2017; labels before 31 Dec 2016", f"{counts['train']:,}"],
            ["Validation", "1 Jan-30 Apr 2017; labels before 30 Apr 2017", f"{counts['validation']:,}"],
            ["Final test", "1 May-31 Aug 2017; evaluation reserved", f"{counts['test']:,}"],
            ["Outside risk set", "Already resolved, ambiguous day, or LeadTime=0", f"{counts['ineligible_at_snapshot']:,}"],
            ["Purged labels", "Training / validation outcomes unavailable at cutoff", f"{counts['purged_train_late_label']} / {counts['purged_validation_late_label']}"],
        ], [80, WIDTH-140, 60]),
        ("image", "arrival_cohorts.png", WIDTH, 158),
        ("p", "The manifest assigns every source row a role. Predictor-group overlap across training, validation, and test is zero. Source-level integrity checks can inspect structure; supervised target patterns use training data only. No final-test predictions, loss values, class rates, or model comparisons are computed."),
        ("p", "Limitations: activity is reconstructed from terminal dates, assuming no terminal-status reversal. Purging can remove longer-running outcomes near boundaries. This is an offline arrival-cohort evaluation, not a fully rolling production simulation. Missing booking/customer IDs prevent a complete independence audit."),
    ])

    page("03 / Features and preprocessing", "EXPLICIT ALLOWLISTS; NO FITTED STATISTICS FROM VALIDATION OR TEST", [
        ("table", ["Block", "Implementation and reason"], [
            ["Supervised numeric", "LeadTime, arrival year/day, weekend/weekday nights, Adults, Children, Babies, IsRepeatedGuest, TotalNights, WeekendShare, and cyclic month sine/cosine."],
            ["Supervised categorical", "Meal, Country, MarketSegment, DistributionChannel, ReservedRoomType, Agent, Company, CustomerType. IDs are categories rather than numeric magnitudes."],
            ["Missing values", "Numeric median from training only; categorical UNKNOWN for true missingness. Agent/Company NOT_APPLICABLE remains its own category."],
            ["Encoding / scaling", "One-hot encoding with unseen categories ignored; StandardScaler after numeric imputation. Fitted within a ColumnTransformer and Pipeline."],
            ["Feature engineering", "TotalNights is booked weekday plus weekend nights; WeekendShare is weekend/total, defined as 0 for zero-night bookings. Month uses sin/cos. These operations are deterministic."],
            ["Clustering", "log1p LeadTime and TotalNights; Adults, Children, Babies, WeekendShare; one-hot MarketSegment, CustomerType, ReservedRoomType. Separate fitted preprocessing on full H2."],
        ], [110, WIDTH-110]),
        ("h", "Conservative exclusions"),
        ("p", "Never supply ADR or IsCanceled as inputs. Remove final status/date, eventual room assignment, changes accumulated through check-in/cancellation, deposits derived from later transactions, waiting-list duration, parking/request aggregates, and prior-booking outcome counts whose as-of timing cannot be checked. Omit redundant arrival week number."),
        ("p", "This policy is task-specific and deliberately conservative. Some excluded fields could be legitimate with timestamp evidence. The complete 31-column decision table, including variable meanings and timing assumptions, is in docs/leakage-audit.md."),
        ("h", "Outliers and skew"),
        ("p", "Training distributions and IQR flags are exported. No percentile winsorization or target-driven row removal is fitted in this baseline. Strongly skewed lead time and stay length use log1p only in clustering, where raw distances would otherwise dominate. Standardized rare counts can still affect geometry and require sensitivity checks."),
    ])

    page("04 / Training exploration", "DISTRIBUTIONS, CATEGORIES, AND UNUSUAL VALUES", [
        ("p", f"Training has {counts['train']:,} eligible records. Its late-cancellation rate is {eda['training_cancellation_rate']:.2%}. ADR has mean {eda['training_adr']['mean']:.2f}, median {eda['training_adr']['median']:.2f}, range {eda['training_adr']['min']:.0f}-{eda['training_adr']['max']:.0f}, and skewness {eda['training_adr']['skewness']:.3f}. There are {eda['training_adr']['zero_count']} zero-rate training records; these may be complimentary or exceptional bookings and are retained."),
        ("image", "training_distributions.png", WIDTH, 171),
        ("p", "Histograms display values through each training 99th percentile for legibility; every tail observation remains in model fitting and evaluation. Lead time and stay duration are right-skewed. IQR rules flag many rare binary/count values and should not be mistaken for evidence that those records are invalid."),
        ("image", "training_categories.png", WIDTH, 171),
        ("p", "The top categories show concentration in particular booking channels and room codes. One-hot encoding avoids imposing false numeric order. Sparse Agent/Company IDs and unseen categories can reduce generalization; model comparisons should test robustness to removing those identifiers."),
        ("p", "Additional exported evidence: a training-only Pearson correlation heatmap, numeric percentiles/ranges, IQR bounds and skewness, every training category count, and the full-source schema/cardinality table. Correlations are descriptive rather than causal and do not establish feature timing."),
    ])

    page("05 / ADR regression baseline", "VALIDATION RESULTS; NO HYPERPARAMETER SEARCH", [
        ("p", "Fit Ridge with alpha=1 and the lsqr solver. Standardize numeric features and one-hot encode categorical features inside the training-fitted pipeline. A training-mean DummyRegressor provides a reference. The recorded ADR is the target; no cancellation or final-status field is supplied."),
        ("table", ["Model", "MAE", "MSE", "RMSE", "R-squared", "Fit seconds"], [
            ["Training mean", f"{mean['mae']:.3f}", f"{mean['mse']:.3f}", f"{mean['rmse']:.3f}", f"{mean['r2']:.3f}", f"{mean['fit_seconds']:.3f}"],
            ["Ridge", f"{ridge['mae']:.3f}", f"{ridge['mse']:.3f}", f"{ridge['rmse']:.3f}", f"{ridge['r2']:.3f}", f"{ridge['fit_seconds']:.3f}"],
        ], [100, 65, 80, 75, 95, WIDTH-415]),
        ("image", "regression_validation.png", 390, 279),
        ("p", f"Relative to the mean reference, validation MAE falls by {(1-ridge['mae']/mean['mae']):.1%} and RMSE falls by {(1-ridge['rmse']/mean['rmse']):.1%}. R-squared {ridge['r2']:.3f} indicates useful but incomplete explanatory performance on this arrival cohort. Larger rate errors remain visible; one linear model is not sufficient for final selection."),
        ("p", "These are validation metrics on 6,545 records, not final-test performance. ADR remains in source rate units; no unsupported currency conversion is imposed. Unconstrained Ridge can produce nonphysical negative rates for unfamiliar inputs, so it is a preliminary model rather than a production pricing tool."),
        ("p", "Milestone 2 will compare additional model families before and after tuning, using temporally ordered training folds and a stated selection criterion. Fit times are measured on the current machine with two numerical threads and are not hardware-independent benchmarks."),
    ])

    page("06 / Cancellation baseline", "RARE LATE CANCELLATIONS MAKE ACCURACY MISLEADING", [
        ("p", "Untuned logistic regression uses C=1, lbfgs, no class weighting, and a fixed 0.5 decision threshold. max_iter=2000 is a convergence allowance rather than a searched hyperparameter. The reference predictor uses the training class prior."),
        ("table", ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "Avg. precision"], [
            ["Prior", f"{prior['accuracy']:.4f}", "0.0000", "0.0000", "0.0000", "0.5000", f"{prior['average_precision']:.4f}"],
            ["Logistic", f"{logit['accuracy']:.4f}", f"{logit['precision']:.4f}", f"{logit['recall']:.4f}", f"{logit['f1']:.4f}", f"{logit['roc_auc']:.4f}", f"{logit['average_precision']:.4f}"],
        ], [68, 70, 70, 65, 60, 78, WIDTH-411]),
        ("image", "classification_validation.png", WIDTH, 163),
        ("table", ["Validation outcome", "Not canceled prediction", "Canceled prediction"], [
            ["Actually not canceled", "6,366 true negatives", "9 false positives"],
            ["Actually late canceled / no-show", "169 false negatives", "1 true positive"],
        ], [200, 155, WIDTH-355]),
        ("p", f"Validation contains 170 positives among 6,545 records ({run['validation_cancellation_rate']:.2%}). At 0.5, the model catches only 1/170 positives; it is <b>not operationally useful at this threshold</b>. Its accuracy is slightly lower than the always-negative reference, illustrating why accuracy alone is unsuitable."),
        ("p", f"ROC-AUC {logit['roc_auc']:.3f} and average precision {logit['average_precision']:.4f}, compared with prevalence {run['validation_cancellation_rate']:.4f}, show some ranking signal. Average precision is the non-interpolated precision-recall summary from scikit-learn, not trapezoidal PR area. Ranking does not make the default decisions adequate."),
        ("p", "Next: evaluate class weighting, nonlinear models, probability calibration, and thresholds using training-fold predictions. Select by a stated cost or precision/recall objective. Do not optimize a threshold against the final test set, or claim this risk-set result covers earlier cancellations."),
    ])

    page("07 / Unsupervised segmentation", "FULL-H2 K-MEANS WITH NO OUTCOME FIELDS", [
        ("p", f"Fit a separate K-Means pipeline on all {cluster['rows']:,} H2 bookings using the booking-profile allowlist. k=3 is a fixed exploratory starting point to inspect a small number of interpretable profiles; it is not the optimal selected k. Ten initializations and seed 382 reduce initialization dependence without a parameter search."),
        ("table", ["Measure", "Observed result"], [
            ["Input representation", f"{cluster['representation_dimensions']} scaled numeric / one-hot dimensions"],
            ["Silhouette", f"{cluster['silhouette']:.4f}; reproducible random sample of 2,000 records"],
            ["Davies-Bouldin", f"{cluster['davies_bouldin']:.4f}; all records"],
            ["Inertia / fit seconds", f"{cluster['inertia']:,.2f} / {cluster['fit_seconds']:.3f}"],
            ["2-D PCA variance", f"{sum(cluster['plot_pca_explained_variance']):.2%} combined; visualization only"],
        ], [175, WIDTH-175]),
        ("image", "clustering_baseline.png", WIDTH, 192),
        ("table", ["Cluster", "Rows", "Median lead", "Median nights", "Mean children"], [
            [str(int(r.cluster)), f"{int(r.records):,}", f"{r.median_lead_days:.0f} days", f"{r.median_nights:.0f}", f"{r.mean_children:.2f}"]
            for r in profiles.itertuples()], [70, 90, 110, 110, WIDTH-380]),
        ("p", "Cluster 0 is a shorter-lead, shorter-stay profile; cluster 1 has longer booking lead times; cluster 2 concentrates bookings with children. These are descriptive booking groups, not verified customer identities or causal segments. Majority market/customer/room labels are similar across groups; numeric party/stay traits drive much of the separation."),
        ("p", "Silhouette around 0.22 suggests overlapping rather than cleanly separated groups. The PCA projection retains only about a third of the representation variance and can hide structure. Rare-count scaling, weighting, alternative k, and stability need systematic investigation before selecting final segments."),
    ])

    page("08 / Leakage audit and verification", "WHAT WAS CHECKED AND WHAT THE CSV CANNOT PROVE", [
        ("table", ["Risk", "Control / evidence"], [
            ["Target / outcome leakage", "Explicit allowlists exclude both targets, final status/date, and selected lifecycle aggregates. A full decision table covers all 31 original columns."],
            ["Snapshot mismatch", "Use the paper's prior-day snapshot and an active-booking risk set. Earlier canceled bookings are excluded rather than relabeled as future outcomes."],
            ["Unavailable training labels", "Purge labels dated at or after the first next-period snapshot day; remove 247 training and 270 validation rows."],
            ["Shared preprocessing", "Supervised imputation, scaling, and encoding fit only on training rows. Serialized imputer statistics match training medians."],
            ["Duplicate contamination", "Same-date identical predictor groups stay within cohorts; all three cross-partition overlap counts equal zero."],
            ["Future information in clustering", "Full-source unsupervised transforms, PCA, and assignments are isolated; none becomes a supervised input or model-selection signal."],
            ["Test reuse", "No supervised test scoring command is supplied in this milestone. The experiment summary explicitly records final_test_evaluated=false."],
            ["Serialization / reproducibility", "Reloaded model predictions match validation reference outputs; the split manifest and source checksum reproduce."],
        ], [145, WIDTH-145]),
        ("p", "Automated tests also cover snapshot-day ambiguity, LeadTime=0, label-availability purging, outcome exclusion, validation-only unknown categories, train-statistic isolation, and zero-night feature behavior. Run main.py verify after baseline and python -m unittest discover -s tests -v."),
        ("p", "Remaining limits: the CSV lacks timestamped field histories, booking IDs, and customer IDs; terminal reversals cannot be audited. Retained snapshot provenance remains an assumption. Cohort/seasonal shift and boundary censoring can affect performance. The rare-positive classification problem requires more suitable decision policies."),
        ("p", "See docs/leakage-audit.md for the complete preliminary audit. A passing code check supports the recorded protocol; it does not certify every unobserved source timestamp or guarantee future generalization."),
    ])

    page("09 / Reproduction and remaining work", "TEAM ROSTER CONFIRMED; GITHUB ACCOUNTS AND CONTRIBUTIONS PENDING", [
        ("h", "Exact reproduction sequence"),
        ("code", "py -3.12 -m venv .venv\n.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt\n.\\.venv\\Scripts\\python.exe main.py download\n.\\.venv\\Scripts\\python.exe main.py baseline\n.\\.venv\\Scripts\\python.exe main.py verify\n.\\.venv\\Scripts\\python.exe main.py report\n.\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v"),
        ("p", f"Verified Python {run['python']}; seed 382; two numerical threads. Pinned packages include scikit-learn {run['packages']['scikit-learn']}, pandas {run['packages']['pandas']}, NumPy {run['packages']['numpy']}, and Matplotlib {run['packages']['matplotlib']}. Exact model parameters and measured fit times are in results/run_summary.json. Artifacts are generated locally; inference loads them without retraining."),
        ("h", "Milestone 2 plan"),
        ("p", "Compare at least three model/algorithm families per task, with baseline and tuned results for each. Add Random Forest and a kernel experiment; document search spaces and purged temporal folds. Prioritize rare-positive average precision and threshold tradeoffs. Compare K-Means, hierarchical clustering, and DBSCAN using unsupervised criteria and stability."),
        ("p", "Complete supervised PCA comparisons for at least two models and an original/PCA/LDA classifier comparison. Freeze selection before final supervised test evaluation. Extend the CLI or add a GUI, expose comparison/tuning/clustering results, and retain saved inference artifacts for the discussion. Detailed plan: docs/milestone-2-plan.md."),
        ("h", "Team and contribution record"),
        ("p", "The six work packages combine technical ownership, analysis, verification, and a report section. Omar Medhat owns preprocessing/splits/leakage; Ahmed Medhat data quality/cleaning/EDA; Ahmed Adel the regression workflow and error analysis; Mohamed Hassan the classification workflow and imbalance analysis; Ahmed Mokhtar clustering/PCA/interpretation; Hala Eltaweel CLI integration, system checks, report assembly and packaging. Each owner verifies their section; Hala integrates the sections. Concrete tasks and handoffs are in docs/contributions.md. These assignments require actual work and evidence from each member's account. All members must understand the complete workflow."),
        ("p", "The ZIP excludes raw downloaded data, fitted models/pipelines, virtual environments, and Git internals. Its README records the repository URL and submitted commit SHA. This remains a technical draft until other GitHub accounts and actual contribution records are completed."),
        ("h", "References"),
        ("small", "[1] Antonio, Almeida and Nunes (2019). Hotel booking demand datasets. Data in Brief 22, 41-49. https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/ . Source semantics: Table 1 and sections 2-3. [2] scikit-learn documentation: https://scikit-learn.org/stable/modules/compose.html and https://scikit-learn.org/stable/modules/clustering.html . Course requirements: supplied CSE382 project brief, Milestone 1 on page 10."),
    ])

    _render(sections, team)


def _render(sections, team):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyReport", fontName="Helvetica", fontSize=9.2, leading=13.0,
                              textColor=NAVY, spaceAfter=9, splitLongWords=True))
    styles.add(ParagraphStyle(name="TableReport", parent=styles["BodyReport"], fontSize=8.1,
                              leading=11, spaceAfter=0))
    styles.add(ParagraphStyle(name="SmallReport", parent=styles["BodyReport"], fontSize=7.7, leading=10))
    styles.add(ParagraphStyle(name="TitleReport", fontName="Helvetica-Bold", fontSize=24, leading=29,
                              textColor=NAVY, spaceAfter=10))
    styles.add(ParagraphStyle(name="KickerReport", fontName="Helvetica-Bold", fontSize=8, leading=11,
                              textColor=TEAL, spaceAfter=10))
    styles.add(ParagraphStyle(name="SubReport", fontName="Helvetica-Bold", fontSize=11.4, leading=15,
                              textColor=TEAL, spaceAfter=6, spaceBefore=6))
    styles.add(ParagraphStyle(name="CodeReport", fontName="Courier", fontSize=7.6, leading=11,
                              textColor=NAVY, backColor=LIGHT, borderPadding=8, spaceAfter=12))
    story, markdown = [], []
    output = ROOT / "reports"
    output.mkdir(exist_ok=True)

    def plain(text):
        text = re.sub(r"<link[^>]*>(.*?)</link>", r"\1", text)
        text = text.replace("<b>", "**").replace("</b>", "**")
        return html.unescape(re.sub(r"<[^>]+>", "", text))

    for index, (title, subtitle, blocks) in enumerate(sections):
        if index:
            story.append(PageBreak())
        story += [Paragraph(subtitle, styles["KickerReport"]), Paragraph(title, styles["TitleReport"])]
        markdown += [f"## {title}", "", subtitle, ""]
        for kind, *content in blocks:
            if kind in ("p", "small", "h"):
                style = {"p": "BodyReport", "small": "SmallReport", "h": "SubReport"}[kind]
                story.append(Paragraph(content[0], styles[style]))
                markdown += [("### " if kind == "h" else "") + plain(content[0]), ""]
            elif kind == "table":
                header, rows, widths = content
                cells = [[Paragraph(html.escape(str(value)), styles["TableReport"]) for value in row]
                         for row in [header, *rows]]
                table = Table(cells, colWidths=widths, hAlign="LEFT", repeatRows=1)
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                    ("LINEBELOW", (0, 0), (-1, 0), 1.1, TEAL),
                    ("LINEBELOW", (0, 1), (-1, -1), .35, colors.HexColor("#DDE5E8")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]))
                story += [table, Spacer(1, 10)]
                markdown += ["| " + " | ".join(map(str, header)) + " |",
                             "| " + " | ".join(["---"]*len(header)) + " |"]
                markdown += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
                markdown.append("")
            elif kind == "image":
                filename, width, height = content
                image = Image(str(ROOT / "results/figures" / filename), width=width, height=height, kind="proportional")
                image.hAlign = "CENTER"
                story += [image, Spacer(1, 9)]
                markdown += [f"![{filename.replace('_', ' ').removesuffix('.png')}](../results/figures/{filename})", ""]
            elif kind == "code":
                escaped = html.escape(content[0]).replace("\n", "<br/>")
                story.append(Paragraph(escaped, styles["CodeReport"]))
                markdown += ["```powershell", content[0], "```", ""]
    (output / "Milestone1_Progress_Report.md").write_text("\n".join(markdown), encoding="utf-8")

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(.6)
        canvas.line(42, 38, A4[0]-42, 38)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(42, 25, "CSE382 | Milestone 1 | Draft - GitHub / contribution details pending")
        canvas.drawRightString(A4[0]-42, 25, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output / "Milestone1_Progress_Report.pdf"), pagesize=A4,
                            leftMargin=42, rightMargin=42, topMargin=40, bottomMargin=52,
                            title="CSE382 Milestone 1 - City Hotel Booking Demand",
                            author=", ".join(member["name"] for member in team["members"]),
                            subject="Data preparation and untuned baselines")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
