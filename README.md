# 🏡 House Prices - Exploratory Data Analysis (Enhanced)

## Project Overview
This enhanced EDA project explores the Housing dataset to identify factors influencing house prices. It now includes advanced data preprocessing, feature engineering, statistical analysis, machine learning models, and interactive visualizations.

## Project Structure

```
House_Prices_EDA/
├── dataset/
│   └── Housing.csv
├── images/
│   ├── correlation_heatmap.png
│   ├── enhanced_correlation_heatmap.png
│   ├── price_furnishing.png
│   ├── price_furnishing_interactive.html
│   ├── price_distribution.png
│   ├── price_mainroad.png
│   ├── 3d_price_analysis.html
│   ├── price_per_sqft_analysis.html
│   └── feature_importance.html
├── Housing_Dataset.py
├── requirements.txt
└── README.md
```

## Key Features & Improvements
- **Advanced Preprocessing:** Outlier removal, feature scaling, and new feature creation (price per sqft, total rooms, amenities count)
- **Statistical Analysis:** ANOVA tests for categorical features
- **Machine Learning Models:** Linear Regression and Random Forest for price prediction
- **Feature Importance:** Random Forest feature importance visualized
- **Interactive Visualizations:** 3D scatter plots, box plots, and feature importance using Plotly
- **Comprehensive Reporting:** Model performance metrics (R², MSE, MAE, cross-validation)

## Instructions to Run Locally

1. Clone the repository or download the files.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the script:
```bash
python Housing_Dataset.py
```
4. View the generated plots and interactive HTML files in the `images/` directory.

## Outputs
- **Analysis Summary:** Printed in the terminal after running the script
- **Statistical Test Results:** p-values for all categorical features
- **Model Performance:** R², MSE, MAE, and cross-validation scores for both models
- **Visualizations:**
  - Enhanced correlation heatmap (`images/enhanced_correlation_heatmap.png`)
  - Interactive price distribution by furnishing status (`images/price_furnishing_interactive.html`)
  - 3D price analysis (`images/3d_price_analysis.html`)
  - Price per square foot analysis (`images/price_per_sqft_analysis.html`)
  - Feature importance (`images/feature_importance.html`)

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Plotly
- SciPy

## License
See [LICENSE](LICENSE) for details.

---

## 🚀 Future Work
- Hyperparameter tuning for machine learning models
- Add more advanced regression models
- Incorporate external data (e.g., economic indicators)
- Build a web dashboard for interactive EDA
