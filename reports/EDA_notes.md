#Data type wrong
    #Data is as object have to convert it to DateTime
#There is one duplicate row in the dataset.Remove it 
#There are rows saying landsize is 0 and building area is positive. This is not error that mean they own the building but not the land. Keep as it is
#There are building area values that are 0. This is error convert these to NaN. Impute using KNNImputer, more than 50% is missing
#Year build min and max has unusual values - Define a range from 1800 to current date. Outer of this range convert to NaN. Impute using KNNImputer more than 50% is missing.
#There are some properties with 0 bathrooms. Convert them to NaN - In preprocessing convert before 1900 to 0 and after 1900 impute with KNNImpute
#The Redundant Feature (Bedroom2 vs Rooms)
#There are lot of missing values in some columns.
    #The target feature (Price) have 7609 missing values. We have to drop them becuase we cant guess(impute) them
    #Remove the missing values row of Postcode
    #CouncilArea, Regionname, Propertycount, Distance even these have less number of missing values, decided to guess(impute) them instead of dropping - Mean Median Impute
    Car - KNNImpute
    Landsize - KNNImpute
    Latitude and Longitude - Suburb Centroid Imputation, Do this after the split 
#Missing values analysis 
    #Price - Random missingness
    #Bedroom2, Bathroom, Car, Landsize - Structural missingness (These columns go missing together)
    #BuildingArea, YearBuilt - Structural missingness (Large blocks of white space (>50% missing))
    #Lattitude, Longtitude - Structural missingness (Identical missingness)
#Inconsistency Scan 
    #In suburb there are some case inconsistencies like 'Croydon' vs 'croydon', 'Viewbank' vs 'viewbank'
        #lowercase the feature 
    #In SellerG there are some dupicates, like these 'Buxton' vs 'Buxton/Advantage' vs 'Buxton/Buxton'. And there are some case inconsistencies
        #have to remove text after / and have to lower case the feature 
#Cardinality check
    #One hot encoding - Method, Regionname, Type
    #Target encoding - CouncilArea, Suburb, SellerG
    #Drop the Address column as it has too many unique values - Extreme Cardinality. Every row is unique. It acts like an ID number and has no predictive power for a general model.
#Target Variable Analysis
    #Apply Log Transformation for target
    #Target Outlier Detection - There are too many outliers. If we drop them that will affect the models accuracy. 
        #When we apply the LOg tranform it solve this outlier problem also 
#Numeric Feature Distribution
        #YearBuilt - Apply StandardScaler afterwards
        #Rooms - Skewness: 0.56 (Symmetrical) - no need to handle skewness, Outliers: 33, Outliers are no error, Rooms more than 7 is okay because there can be properties that can have 16 rooms (No need to do anything to this feature).No need to handle outliers. Apply RobustScaler
        #Distance - Skew: 1.5036 (Right skewed) - Do The Log Transform / must use np.log1p, because the min is 0. Outliers: 947 - Dont remove outliers, instead use RobustScaler
        #Bedroom2 - Drop this column 
        #Bathroom - Skew: 1.3797 (Right skewed) - Do the Log Transform / must use np.log, because the min is not 0. Outliers: 204 - Dont remove outliers,Apply RobustScaling
        #Car - Skew:1.6829 (Right skewed) - Do the Log Transform / must use np.log1p, because the min is 0. Outliers: 954 - Dont remove outliers, instead apply RobustScaler
        #Landsize - Skew: 40.3815 - (Right skewed) - Do the Log Transform / must use np.log1p, because the min is 0. Outliers: 365 - Max: 146699.0 - Apply RobustScaler
        #BuildingArea - Skew: 87.0791 - (Right skewed) - Do the Log Transform / must use np.log, because the min is not 0. Outliers: 624 - Max: 44515.0 - Apply RobustScaler
        #Lattitude & Longtitude - Keep as it is.No need to handle outliers or skewness. Apply StandardScaler to normalize the range for the model.
        #Propertycount - 1.0159 - Skew: 1.0159 - Do the Log Transform / must use np.log, because the min is not 0. Outliers: 591 - Max: 21650.0 Use Robust Scaling
    #Safety Tip: In automated pipelines, it is safer to ALWAYS use np.log1p (Log of 1+x) instead of np.log. Even if Bathroom min is 1.0 now, if your Test Set (or future data) has a house with 0 Bathrooms (e.g., a studio or unfinished house), np.log(0) will crash your code (return -infinity).np.log1p(0) returns 0. It is bulletproof. Just use np.log1p for everything.
#Categorical Feature Distribution
    #Suburb - Target Encoding
    #Type - Keep as it is. One hot encoding
    #SellerG - Target Encoding
    #Regionname - Group these categories into Regional Victoria - 'Eastern Victoria', 'Northern Victoria', 'Western Victoria'. After that apply one hot encoding.
#Correlation and Relationship Analysis
    #Numeric vs Numeric (Correlation)
        #Drop Bedroom2 feature, its redundant with Rooms feature 
    #Categorical vs Categorical
        #Drop the CouncilArea feature it is redundant
    #Numeric vs Target 
        #As you can see in Distance vs Price plot, the price of house is decreasing when the distance increasing and the price of a unit is not effect it is flat line.
    #Categorical vs Target
        #In type vs price you can see house has effect price more than unit and townhouse
        #In Regionname vs price you can see the price increase like staircase with the regions

#Features to remove 
    #Address - Extereme cardinality
    #Bedroom 2 - Redundant with rooms feature 
    #CouncilArea - redundant 
    #Date - Extract Year: Create a new column called Sold_Year. This    captures the "Inflation Effect."
            Drop Date: Remove the original column because the specific day doesn't matter and it's messy text.
    #Method - It contains future information. Although it is statistically powerful, it is logically invalid for a prediction model. It describes the result, not the property
    #Postcode - The heatmap shows it is identical to Suburb
#Handeling missing values (Imputation)
    #BuildingArea - KNNImpute
    #YearBuild - KNNImpute
    #Bathroom - KNNImpute
    #Car - KNNImpute
    Landsize - KNNImpute
    #Latitude and Longitude - Suburb Centroid Imputation
    #Regionname, Propertycount - Mean /Median/ Mode Impute

#Rare Category Grouping
    #Regionname - Group these categories into Regional Victoria - 'Eastern Victoria', 'Northern Victoria', 'Western Victoria'.

#Log Transformation
    #Target Log Transform - Apply Log Transformation for target
    #Distance - Skew: 1.5036 (Right skewed) - Do The Log Transform / must use np.log1p, because the min is 0. 
    #Bathroom - Skew: 1.3797 (Right skewed) - Do the Log Transform / must use np.log, because the min is not 0. 
    #Car - Skew:1.6829 (Right skewed) - Do the Log Transform / must use np.log1p, because the min is 0. 
    #Landsize - Skew: 40.3815 - (Right skewed) - Do the Log Transform / must use np.log1p, because the min is 0. 
    #BuildingArea - Skew: 87.0791 - (Right skewed) - Do the Log Transform / must use np.log, because the min is not 0. 
    #Propertycount - 1.0159 - Skew: 1.0159 - Do the Log Transform / must use np.log, because the min is not 0. Outliers: 591 - Max: 21650.0 
    #Safety Tip: In automated pipelines, it is safer to ALWAYS use np.log1p (Log of 1+x) instead of np.log. Even if Bathroom min is 1.0 now, if your Test Set (or future data) has a house with 0 Bathrooms (e.g., a studio or unfinished house), np.log(0) will crash your code (return -infinity).np.log1p(0) returns 0. It is bulletproof. Just use np.log1p for everything.

#Scaling 
    #YearBuilt - Apply StandardScaler
    #Rooms - Skewness: 0.56 (Symmetrical) - no need to handle skewness, Outliers: 33, Outliers are no error, Rooms more than 7 is okay because there can be properties that can have 16 rooms (No need to do anything to this feature).No need to handle outliers. Apply RobustScaler
    #Distance - Outliers: 947 - Dont remove outliers, instead use RobustScaler
    #Bathroom - Outliers: 204 - Dont remove outliers,Apply RobustScaling
    #Car - Outliers: 954 - Dont remove outliers, instead apply RobustScaler
    #Landsize - Outliers: 365 - Max: 146699.0 - Apply RobustScaler
    #BuildingArea - Outliers: 624 - Max: 44515.0 - Apply RobustScaler
    #Lattitude & Longtitude - Keep as it is.No need to handle outliers or skewness. Apply StandardScaler to normalize the range for the model.
    #Propertycount - Use Robust Scaling

#Encoding 
    #Type - One hot encoding
    #SellerG - Target Encoding
    #Regionname - One hot encoding
    #Suburb - Target Encoding





