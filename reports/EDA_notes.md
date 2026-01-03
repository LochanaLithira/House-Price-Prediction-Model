# EDA Notes & Preprocessing Strategy

## 1. Data Cleaning & Inconsistency Handling
*   **Data Type Conversion**: Convert `Date` column to DateTime object.
*   **Duplicates**: Remove the single duplicate row found in the dataset.
*   **Logic Errors**:
    *   **Landsize = 0**: Keep as is (implies unit/apartment ownership without land).
    *   **BuildingArea = 0**: Convert to `NaN` (error).
    *   **YearBuilt**: Define valid range (1800 - Current Date). Convert values outside this range to `NaN`.
    *   **Bathrooms = 0**: Convert to `NaN` (likely error). *Preprocessing Note: For records before 1900, set to 0; after 1900, impute.*
*   **Text Inconsistencies**:
    *   **Suburb**: Fix case inconsistencies (e.g., 'Croydon' vs 'croydon') -> Lowercase.
    *   **SellerG**: Fix duplicates and case issues (e.g., 'Buxton' vs 'Buxton/Advantage') -> Remove text after `/` and lowercase.

## 2. Feature Selection (Columns to Drop)
*   **Address**: Extreme cardinality (acts as ID).
*   **Bedroom2**: Redundant with `Rooms`.
*   **CouncilArea**: Redundant (highly correlated with other location features).
*   **Date**: Drop after extracting `Sold_Year`.
*   **Method**: Contains future information (describes result, not property).
*   **Postcode**: Redundant (identical to `Suburb` in heatmap).
*   **Rows with Missing Target**: Drop rows where `Price` is missing (7609 rows).
*   **Rows with Missing Postcode**: Remove rows with missing `Postcode`.

## 3. Missing Value Imputation
*   **Price**: Drop rows (Target variable).
*   **Postcode**: Drop rows.
*   **KNN Imputation**:
    *   `Car`
    *   `Landsize`
    *   `BuildingArea` (>50% missing)
    *   `YearBuilt` (>50% missing)
    *   `Bathroom`
*   **Suburb Centroid Imputation**:
    *   `Latitude`
    *   `Longitude` (Do this after train/test split).
*   **Mean/Median/Mode Imputation**:
    *   `Regionname`
    *   `Propertycount`
    *   `Distance`

## 4. Feature Engineering
*   **Date**: Extract `Year` to create `Sold_Year` (captures inflation effect).
*   **Regionname**: Group rare categories ('Eastern Victoria', 'Northern Victoria', 'Western Victoria') into a single category: `Regional Victoria`.

## 5. Transformations (Skewness Handling)
*   **Strategy**: Use `np.log1p` for all log transformations to handle 0 values safely.
*   **Apply Log Transform (`np.log1p`)**:
    *   `Target` (Price)
    *   `Distance` (Skew: 1.50)
    *   `Bathroom` (Skew: 1.38)
    *   `Car` (Skew: 1.68)
    *   `Landsize` (Skew: 40.38)
    *   `BuildingArea` (Skew: 87.08)
    *   `Propertycount` (Skew: 1.02)
*   **No Transform**:
    *   `Rooms` (Symmetrical, Skew: 0.56)
    *   `Latitude` & `Longitude`

## 6. Scaling (Outlier Handling)
*   **RobustScaler** (Handles outliers well):
    *   `Rooms` (Outliers present but valid)
    *   `Distance`
    *   `Bathroom`
    *   `Car`
    *   `Landsize`
    *   `BuildingArea`
    *   `Propertycount`
*   **StandardScaler**:
    *   `YearBuilt`
    *   `Latitude`
    *   `Longitude`

## 7. Encoding Categorical Features
*   **One-Hot Encoding**:
    *   `Type`
    *   `Regionname` (after grouping)
*   **Target Encoding**:
    *   `Suburb`
    *   `SellerG`

## 8. Analysis Insights
*   **Missingness Patterns**:
    *   `Price`: Random missingness.
    *   `Bedroom2`, `Bathroom`, `Car`, `Landsize`: Structural missingness (missing together).
    *   `BuildingArea`, `YearBuilt`: Structural missingness (>50% missing).
    *   `Latitude`, `Longitude`: Identical missingness.
*   **Correlations**:
    *   `Distance` vs `Price`: Negative correlation for houses; flat for units.
    *   `Type` vs `Price`: Houses affect price more than units/townhouses.
    *   `Regionname` vs `Price`: Step-like price increase across regions.





