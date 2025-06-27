# SDG 13: Climate Action — Predicting CO₂ Emissions with Machine Learning

## Project Summary
This project addresses the United Nations Sustainable Development Goal 13 (Climate Action) by building a machine learning model to predict annual CO₂ emissions for the United States. Accurate forecasting of emissions can help policymakers and researchers design effective interventions to combat climate change.

## Problem Statement
Climate change, driven largely by greenhouse gas emissions, is one of the most pressing global challenges. Predicting future CO₂ emissions enables better planning and policy-making to reduce environmental impact and meet international climate targets.

## Machine Learning Approach
- **Type:** Supervised Learning (Regression)
- **Algorithm:** Random Forest Regressor
- **Features Used:**
  - Year
  - GDP
  - Population
  - Energy per Capita
  - Primary Energy Consumption
  - CO₂ per Capita
  - Coal CO₂
  - Oil CO₂
  - Gas CO₂
- **Target Variable:** Annual CO₂ emissions (metric tons)
- **Dataset:** [Our World in Data - CO₂ and Greenhouse Gas Emissions](https://www.kaggle.com/datasets/danielbeltschneider/co2-and-greenhouse-gas-emissions)

## Results
- **Mean Absolute Error (MAE):** 65.81
- **Root Mean Squared Error (RMSE):** 82.38
- **R² Score:** 0.98

The model explains 98% of the variance in actual CO₂ emissions for the United States, demonstrating excellent predictive power. The low error values indicate that the model's predictions are very close to the real data.

## Ethical & Social Considerations
- **Data Bias:** The dataset may have missing years or inconsistencies in country reporting, which could affect model accuracy.
- **Fairness:** The model is trained on historical data for the United States; results may differ for other countries or regions.
- **Sustainability Impact:** Accurate predictions can inform climate policy, support emission reduction strategies, and promote sustainable development.

## Conclusion
This project demonstrates how machine learning can be leveraged to support SDG 13 by providing accurate, data-driven insights into CO₂ emissions. Such tools are vital for driving effective climate action and building a more sustainable future. 