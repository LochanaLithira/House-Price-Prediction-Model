# House-Price-Prediction-Model

```mermaid
flowchart TD
    A[Start: Raw Dataset] --> B[Before Split]

    B --> B1[1. Data Typing]
    B1 --> B2[2. Detect Similar Columns & Remove Them]
    B2 --> B3[3. Cardinality Check - Diagnostic]
    B3 --> B4[4. Fix Inconsistencies - Standardize Labels]
    B4 --> B5[5. Cardinality Check - Final]
    B5 --> B6[6. Remove Duplicates]
    B6 --> B7[7. Sanity Check]
    B7 --> B8[8. Check Missing Patterns]
    B8 --> B9[9. Handle Structural Missing Values]
    B9 --> B10[10. Date Decomposition]
    B10 --> B11[11. Text Extraction - Optional]
    B11 --> B12[12. Row-Wise Feature Engineering - No Aggregates]
    B12 --> B13[13. Binning - Domain Logic]
    B13 --> C[Split Dataset: Train / Test]

    C --> D[After Split - Train Only]
    D --> D1[1. Check Distribution of Target]
    D1 --> D2[2. Log-transform Target if Skewed]
    D2 --> D3[3. Handle Random Missing Values - Imputation]
    D3 --> D4[4. Handle Outliers]
    D4 --> D5[5. Check Distributions & Transform Features]
    D5 --> D6[6. Aggregate Feature Engineering - Group Stats]
    D6 --> D7[7. Correlation Check Type A - Numeric Only]
    D7 --> D8[8. Feature Encoding]
    D8 --> D9[9. Feature Scaling]
    D9 --> D10[10. Full Correlation & Relationships Check]
    D10 --> D11[11. Feature Selection]
    D11 --> E[Preprocessed Dataset Ready for Modeling]
