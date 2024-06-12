# 🏡 House Prices - Exploratory Data Analysis

##  Project Overview
This EDA project explores the Housing dataset to identify factors influencing house prices. It includes data cleaning, statistical analysis, and insightful visualizations to uncover significant trends and relationships.

##  Project Structure

```
House_Prices_EDA/
├── dataset/
│   └── Housing.csv
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── images/
│   ├── correlation_heatmap.png
│   ├── price_furnishing.png
│   ├── price_distribution.png
│   └── price_mainroad.png
└── requirements.txt
```

##  Key Findings

###  Numerical Insights:
- Strong positive correlation between **Area** and **Price** (`0.54`).
- Houses with more **Bathrooms** (`0.52`), **Stories** (`0.42`), and **Bedrooms** (`0.37`) command higher prices.
- Presence of **Parking** also moderately increases house prices (`0.38`).

###  Categorical Insights:
- Houses located on **Main Roads** and within **Preferred Areas** fetch higher prices.
- Features such as **Guestrooms**, **Basements**, **Hot Water Heating**, and **Air Conditioning** significantly elevate house values.
- **Furnished** houses are notably pricier, followed by semi-furnished and unfurnished.

##  Visualizations:
- **Correlation Heatmap**: Illustrates numeric relationships clearly.
- **Average Price by Furnishing Status**: Clearly shows furnished houses commanding highest prices.
- **Distribution of House Prices**: Highlights the price range distribution.
- **Price Distribution based on Main Road Access**: Indicates a higher median price for houses near main roads.

##  Technologies Used:
- Python
- Pandas
- Matplotlib
- Seaborn

##  Instructions to Run Locally:

1. Clone the repository or download the files.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the notebook:
```bash
jupyter notebook
```

##  Conclusion:
House prices are heavily influenced by features like area, furnishing status, location (main road), and amenities (basement, air conditioning). Understanding these factors clearly can significantly improve predictive modeling efforts.

---

## 🚀 Future Work:
- Implement advanced regression models (Linear Regression, Random Forest) to predict house prices.
- Conduct deeper analysis incorporating external factors (economic indicators, market trends).
