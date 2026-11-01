# E-Commerce Customer Segmentation: RFM Analysis with K-Means & DBSCAN

An end-to-end unsupervised machine learning pipeline to segment online retail customers using **RFM (Recency, Frequency, Monetary)** features. This project benchmarks **K-Means** against **DBSCAN**, applies **PCA** and **t-SNE** for 2D manifold visualization, and translates raw statistical clusters into actionable marketing strategies.

---

## Overview & Business Context

Traditional rule-based segmentation often misses subtle behavioral shifts in purchasing patterns. By analyzing transaction logs over a full annual cycle, this pipeline:
1. Aggregates granular purchase transactions into customer-level RFM metrics.
2. Removes right-tail skewness using a log transform ($\log(1+x)$) and standardizes feature variance.
3. Compares partitioning clustering (**K-Means**) against density-based clustering (**DBSCAN**).
4. Proves cluster separation using linear (**PCA**) and non-linear (**t-SNE**) projections.
5. Maps algorithmic clusters directly to actionable marketing personas (VIPs, Loyalists, Churned, and Wholesale Outliers).

---

## Dataset

* **Source:** [Online Retail Dataset (UCI Machine Learning Repository / Kaggle)](https://www.kaggle.com/datasets/vijayuv/onlineretail)
* **Scope:** 541,909 raw transactions from a UK-based registered online retailer.
* **Cleaning Workflow:**
  * Dropped null customer entries (guest checkouts).
  * Filtered cancellations (`Quantity <= 0`) and pricing errors (`UnitPrice <= 0`).
  * Aggregated by `CustomerID` relative to a reference snapshot date ($T_{\text{max}} + 1\text{ day}$).

---

## Pipeline Architecture
Raw Invoices ──> RFM Aggregation ──> Log1p & StandardScaler
│
┌────────────────────────────┴──────────────────────────┐
▼                                                       ▼
K-Means Clustering                                       DBSCAN Clustering
• Elbow Method (Inertia)                               • k-NN Distance Knee (eps=0.45)
• Silhouette Score (K=4)                               • Anomaly/Noise Isolation (-1)
│                                                       │
└────────────────────────────┬──────────────────────────┘
▼
Dimensionality Reduction
• PCA (Global Variance)
• t-SNE (Local Manifold)
│
▼
Actionable Marketing Personas

---

## Key Experimental Results

### 1. K-Means Diagnostics ($K = 4$)
* **Elbow Curve:** Significant reduction in inertia from $K=2$ ($6,481.2$) through $K=4$ ($3,938.5$), after which the curve flattens[cite: 1].
* **Silhouette Analysis:** Peak stability observed at $K=4$ ($s = 0.3371$) before a steep drop at higher cluster counts[cite: 1].

### 2. DBSCAN Density Mapping (`eps=0.45`, `min_samples=5`)
* **Core Segments:** Captured 2 large clusters representing regular shopping volume ($2,781$ customers) and dormant single-order buyers ($1,482$ customers)[cite: 3].
* **Anomaly Detection (`-1` Noise):** Successfully isolated **70 high-volume wholesale/B2B buyers** ($30.1$ average orders, $\$39,453.90$ average spend) without distorting normal retail cluster centroids[cite: 3].

### 3. Dimensionality Reduction (PCA vs. t-SNE)
* **PCA:** Captures the linear variance spread across PC1 and PC2, showing how K-Means partitions the continuous customer plane[cite: 3].
* **t-SNE:** Resolves non-linear cluster neighborhoods, clearly isolating the DBSCAN noise points along the periphery of the embedding[cite: 3].

---

## Derived Marketing Personas

| Persona | RFM Characteristics | Segment Size | Recommended Strategy |
| :--- | :--- | :--- | :--- |
| **Champions / VIPs** | Low Recency, High Frequency, High Spend | Core Retail | Early access to product drops, dedicated support, loyalty points. Avoid aggressive discounts. |
| **Potential Loyalists** | Low Recency, Moderate Frequency & Spend | Core Retail | Category cross-selling, bundle incentives, and loyalty membership sign-ups. |
| **Hibernating / Churned** | High Recency, Low Frequency ($1$), Low Spend | $1,482$ customers | Automated win-back campaigns, feedback surveys, or suppression from bulk email lists. |
| **B2B / Wholesale Outliers** | DBSCAN Cluster `-1` (High F, Extreme M) | $70$ customers | Direct hand-off to dedicated B2B account managers for net-30 terms and pallet pricing. |

---

## Takeaway: K-Means vs. DBSCAN

* **K-Means** is the preferred tool for **everyday marketing operations**: it neatly divides the continuous customer base into balanced, actionable groups for targeted campaigns[cite: 3].
* **DBSCAN** is the preferred tool for **outlier and fraud detection**: it flags extreme spenders and inactive accounts without manual rule-setting, preventing extreme values from skewing normal customer analytics[cite: 3].

---

## Setup & Reproduction

### Prerequisites
```bash
pip install pandas numpy scikit-learn matplotlib seaborn kaggle
