# Milestone 2 plan

These are planned experiments, not completed results.

1. Review the late-cancellation risk-set definition with the TA. Keep the snapshot-consistent task or obtain earlier timestamped features before claiming booking-creation prediction. Confirm member identities and actual contribution statements.
2. Regression: compare Ridge, Random Forest, and RBF SVR (with a documented computational subset if necessary). Run and retain untuned and tuned results for every family. Use purged temporal validation folds inside the training period; optimize MAE and report MAE, MSE/RMSE, R-squared, runtime, and final untouched-test results after selection.
3. Classification: compare logistic regression, Random Forest, and RBF SVC. Optimize average precision using temporally ordered, label-purged training folds. Explore class weights and decision thresholds on training-fold predictions only; explicitly compare precision/recall tradeoffs. Assess calibration. Do not interpret high accuracy as success with the rare late-cancellation outcome.
4. Clustering: compare baseline and systematically optimized K-Means, agglomerative clustering, and DBSCAN. Explore k=2..10 for K-Means; document a reproducible representative sample and dendrogram for hierarchical clustering; explore DBSCAN eps/min_samples using unsupervised criteria and noise coverage. Use silhouette, Davies-Bouldin, cluster size/stability, and interpretation without supervised labels. Audit feature weighting and rare-child/baby scaling effects.
5. PCA/LDA: fit transformations inside training folds. Compare at least two supervised models with and without PCA. For one classifier compare original, PCA, and LDA representations; binary LDA has at most one discriminant component. Report explained variance and retained-component reasoning. The Milestone 1 cluster PCA plot does not satisfy these supervised comparisons.
6. Search design: document every search space, budget, seed, objective, and best configuration. Include at least one ensemble model. Freeze preprocessing, models, and decision policy before a single final supervised test evaluation.
7. Application: extend the existing CLI with model comparison/tuning views and clustering analysis, or add Streamlit. Inference must load saved pipelines and never retrain implicitly. Include input validation, model metadata, and scope explanations.
8. Final delivery: generate the complete report, baseline-versus-tuned tables, figures, saved artifacts for local discussion, reproducible commands, and ZIP without fitted models/pipelines. Record actual contributions through each member's account. Prepare every member to explain the entire workflow.

Milestone 2 deadline in the supplied brief: 11 December 2026. Final discussions: 18-24 December 2026.
