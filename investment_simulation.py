# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:48:46 2025

@author: my199
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ------------------
# ▶ Streamlit UI
# ------------------
st.title("資産運用シミュレーション")

start_age = st.slider("開始年齢", min_value=20, max_value=60, value=30)
monthly_contribution = st.slider("月次投資額 (万円)", min_value=1, max_value=50, value=5)
equity_ratio = st.slider("株当比率 (残りは債券)", min_value=0.0, max_value=1.0, step=0.05, value=0.5)

retirement_age = 65

# ------------------
# ▶ Market Parameters
# ------------------
equitypremium = 0.055
bondpremium = 0.009

returnYearly = np.array([equitypremium, bondpremium])
volatilityYearly = np.array([0.23, 0.03])
correlation = -0.3
corrYearly = np.array([[1, correlation], [correlation, 1]])

# Monthly conversion
monthly_returns = returnYearly / 12
monthly_volatility = volatilityYearly / np.sqrt(12)
cov_matrix = np.diag(monthly_volatility) @ corrYearly @ np.diag(monthly_volatility)

# Investment Conditions
weights = np.array([equity_ratio, 1 - equity_ratio])
monthly_contribution_value = monthly_contribution
n_simulations = 1000
n_years = retirement_age - start_age
n_months = n_years * 12
ages = np.arange(start_age, retirement_age + 1)

# ------------------
# ▶ Simulation
# ------------------
all_trajectories = np.zeros((n_simulations, n_years + 1))

for i in range(n_simulations):
    portfolio_value = 0
    values_by_year = [portfolio_value]
    returns = np.random.multivariate_normal(monthly_returns, cov_matrix, n_months)
    for month in range(n_months):
        monthly_return = np.dot(weights, returns[month])
        portfolio_value *= (1 + monthly_return)
        portfolio_value += monthly_contribution_value
        if (month + 1) % 12 == 0:
            values_by_year.append(portfolio_value)
    all_trajectories[i, :] = values_by_year

# ------------------
# ▶ Percentiles
# ------------------
final_values = all_trajectories[:, -1]
percentile_25_val = np.percentile(final_values, 25)
percentile_50_val = np.percentile(final_values, 50)
percentile_75_val = np.percentile(final_values, 75)

idx_25 = np.abs(final_values - percentile_25_val).argmin()
idx_50 = np.abs(final_values - percentile_50_val).argmin()
idx_75 = np.abs(final_values - percentile_75_val).argmin()

trajectory_25 = all_trajectories[idx_25]
trajectory_50 = all_trajectories[idx_50]
trajectory_75 = all_trajectories[idx_75]

# ------------------
# ▶ Plotting
# ------------------
st.subheader("投資結果グラフ")
fig, ax = plt.subplots(figsize=(12, 6))

for i in range(n_simulations):
    ax.plot(ages, all_trajectories[i], color='gray', alpha=0.03)

ax.plot(ages, trajectory_75, color='blue', linestyle='dashed', label='75th Percentile')
ax.plot(ages, trajectory_50, color='red', label='50th Percentile')
ax.plot(ages, trajectory_25, color='blue', linestyle='dashed', label='25th Percentile')

# Saving-only scenario
saving_trajectory = monthly_contribution_value * 12 * (ages - start_age)
ax.plot(ages, saving_trajectory, color='green', linewidth=2, label='Saving Only')

ax.set_title('Investment Simulation Until Retirement')
ax.set_xlabel('Age')
ax.set_ylabel('10,000 Yen')
ax.legend()
ax.set_ylim(0, np.max(trajectory_75) * 1.2)

st.pyplot(fig)