import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

# Load datasets
pop = pd.read_csv('Unit04 Global_Population.csv')
gdp = pd.read_csv('Unit04 Global_GDP.csv')

# Automatically detect common year columns between 2001 and 2021
year_cols_pop = {c for c in pop.columns if c.isdigit()}
year_cols_gdp = {c for c in gdp.columns if c.isdigit()}
common_years = sorted(year_cols_pop & year_cols_gdp, key=int)
years = [y for y in common_years if 2001 <= int(y) <= 2021]

# Reshape to long format
pop_long = pop.melt(
    id_vars=['Country Name','Country Code'],
    value_vars=years,
    var_name='Year',
    value_name='Population'
)
gdp_long = gdp.melt(
    id_vars=['Country Name','Country Code'],
    value_vars=years,
    var_name='Year',
    value_name='GDP'
)

# Convert Population and GDP to numeric (remove commas if present)
pop_long['Population'] = (
    pop_long['Population']
    .astype(str)
    .str.replace(',','')
    .replace('', np.nan)
)
pop_long['Population'] = pd.to_numeric(pop_long['Population'], errors='coerce')

gdp_long['GDP'] = (
    gdp_long['GDP']
    .astype(str)
    .str.replace(',','')
    .replace('', np.nan)
)
gdp_long['GDP'] = pd.to_numeric(gdp_long['GDP'], errors='coerce')

# Merge and drop rows with missing values
df = pd.merge(pop_long, gdp_long, on=['Country Name','Country Code','Year']).dropna(subset=['Population','GDP'])

# Compute per‑capita GDP
df['GDP_per_capita'] = df['GDP'] / df['Population']

# Aggregate means per country
summary = df.groupby(['Country Name','Country Code']).agg(
    mean_population=('Population','mean'),
    mean_gdp_per_capita=('GDP_per_capita','mean')
).reset_index()

# Task A: Scatter plot & Pearson correlation
plt.figure()
plt.scatter(summary['mean_population'], summary['mean_gdp_per_capita'])
plt.xlabel('Mean Population (2001–2021)')
plt.ylabel('Mean Per Capita GDP (USD)')
plt.title('Mean Population vs Mean Per Capita GDP')
plt.show()

r, p = pearsonr(summary['mean_population'], summary['mean_gdp_per_capita'])
print(f'Pearson r = {r:.3f}, p-value = {p:.2e}')

# Task B: Linear regression
X = summary[['mean_population']]
y = summary['mean_gdp_per_capita']
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print(f'Slope = {model.coef_[0]:.3e}, Intercept = {model.intercept_:.3e}, R² = {r2:.3f}')

plt.figure()
plt.scatter(summary['mean_population'], summary['mean_gdp_per_capita'], label='Data')
plt.plot(summary['mean_population'], y_pred, linewidth=2, label='Fit')
plt.xlabel('Mean Population (2001–2021)')
plt.ylabel('Mean Per Capita GDP (USD)')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()
