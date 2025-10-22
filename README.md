# BMW Sales Classification (High vs Low)

End‑to‑end EDA, preprocessing, visualization, and modeling for `Sales_Classification` using a CSV with ~10 features such as Model, Year, Region, Color, Fuel_Type, Transmission, Engine_Size_L, Mileage_KM, Price_USD, and Sales_Volume from 2010–2024. [attached_file:18]

## Files
- BMW_Sales_Classification_EDA_Model.ipynb — Complete pipeline: checks, EDA, preprocessing, logistic‑regression baseline, PyTorch MLP, evaluation. 
- BMW-sales-data-2010-2024-1.csv — Dataset placed in the same folder as the notebook before running. 
- README.md — This guide with setup and run instructions for reproducibility. 

## Setup
- Python 3.9+ with packages: pandas, numpy, matplotlib, seaborn, scikit‑learn, torch (CUDA optional). 
- Launch Jupyter (Lab/Notebook) or VS Code, open BMW_Sales_Classification_EDA_Model.ipynb, and run all cells in order. 

## Procedure
- Loads the CSV and audits shape, dtypes, missing values, and duplicates to guide preprocessing and sanity checks. 
- Creates 5+ visuals: numeric histograms/boxplots, correlation heatmap, sampled pairplot, and categorical count plots with target hue. 
- Preprocesses with ColumnTransformer: median impute + StandardScaler for numerics; most‑frequent impute + One‑Hot for categoricals based on the dataset’s header. 
- Splits data with stratified 70/15/15 train/val/test to avoid leakage and preserve class balance for the High/Low label. 
- Trains a balanced Logistic Regression baseline and a compact PyTorch MLP with BatchNorm, Dropout, ReduceLROnPlateau, and early stopping for robust performance. 
- Evaluates with accuracy, precision, recall, F1, ROC–AUC plus confusion matrix and ROC curve on the held‑out test set. 

## How to run
- Put BMW-sales-data-2010-2024-1.csv next to the notebook, open the notebook, and Run All to generate figures and metrics without path edits. 
- If pairplots are slow, reduce the number of plotted numeric columns or the sample size in the EDA cell for faster rendering. 

## Notes
- The target `Sales_Classification` contains exactly two classes High and Low in the provided file; the notebook maps them explicitly to {Low:0, High:1} for consistent metrics and class weighting. 
- Numeric features (e.g., Year, Engine_Size_L, Mileage_KM, Price_USD, Sales_Volume) are scaled, while categoricals (e.g., Model, Region, Color, Fuel_Type, Transmission) are one‑hot encoded per the CSV schema. 
