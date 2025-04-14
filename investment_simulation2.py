# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:48:46 2025

@author: my199
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 初期設定
st.set_page_config(page_title="資産運用シミュレーション", layout="centered")

st.title("📈 資産運用シミュレーション")

# ----------------------------
# 🧾 前提条件の表示
# ----------------------------
with st.expander("📌 前提条件"):
    st.markdown("""
    - *株式の期待リターン*：5.5%
    - *株式のリスク（年率）*：23%
    - *債券の期待リターン*：0.9%
    - *債券のリスク（年率）*：3%
    - *株式と債券の相関*：-0.3
    - *インフレ率*：2%
    """)

# ----------------------------
# 🎯 入力項目
# ----------------------------
st.markdown("### 🔧 シミュレーション設定")

col1, col2, col3 = st.columns(3)

with col1:
    start_age = st.slider("開始年齢", min_value=20, max_value=60, value=30)
with col2:
    monthly_contribution = st.slider("月額積立額（万円）", min_value=1, max_value=30, value=5)
with col3:
    equity_ratio = st.slider("株式比率(残りは債券)（%）", 0, 100, 50)

# 実行ボタン
if st.button("シミュレーションを実行",type = "primary"):

    # ----------------------------
    # 📊 パラメータ設定
    # ----------------------------
    equity_return = 0.055
    bond_return = 0.009
    inflation = 0.02

    real_equity_return = equity_return - inflation
    real_bond_return = bond_return - inflation

    returnYearly = np.array([real_equity_return, real_bond_return])
    volatilityYearly = np.array([0.23, 0.03])
    correlation = -0.3
    corrYearly = np.array([[1, correlation],
                           [correlation, 1]])

    # 月次変換
    monthly_returns = returnYearly / 12
    monthly_volatility = volatilityYearly / np.sqrt(12)
    cov_matrix = np.diag(monthly_volatility) @ corrYearly @ np.diag(monthly_volatility)

    # 投資設定
    weights = np.array([equity_ratio / 100, 1 - (equity_ratio / 100)])
    retirement_age = 65
    n_simulations = 1000
    n_years = retirement_age - start_age
    n_months = n_years * 12
    ages = np.arange(start_age, retirement_age + 1)
    years = np.arange(2025 + (start_age - 30), 2025 + (retirement_age - 30) + 1)

    all_trajectories = np.zeros((n_simulations, n_years + 1))  # 年単位

    for i in range(n_simulations):
        portfolio_value = 0
        values_by_year = [portfolio_value]
        returns = np.random.multivariate_normal(monthly_returns, cov_matrix, n_months)
        for month in range(n_months):
            monthly_return = np.dot(weights, returns[month])
            portfolio_value *= (1 + monthly_return)
            portfolio_value += monthly_contribution
            if (month + 1) % 12 == 0:
                values_by_year.append(portfolio_value)
        all_trajectories[i, :] = values_by_year

    # ----------------------------
    # 📉 パーセンタイル計算
    # ----------------------------
    final_values = all_trajectories[:, -1]
    p25_val = np.percentile(final_values, 25)
    p50_val = np.percentile(final_values, 50)
    p75_val = np.percentile(final_values, 75)

    idx_25 = np.abs(final_values - p25_val).argmin()
    idx_50 = np.abs(final_values - p50_val).argmin()
    idx_75 = np.abs(final_values - p75_val).argmin()

    trajectory_25 = all_trajectories[idx_25]
    trajectory_50 = all_trajectories[idx_50]
    trajectory_75 = all_trajectories[idx_75]

    # ----------------------------
    # 💹 グラフ描画
    # ----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    for i in range(n_simulations):
        ax.plot(ages, all_trajectories[i], color='gray', alpha=0.03)

    ax.plot(ages, trajectory_75, color='blue', linestyle='dashed', linewidth=2, label='75th Percentile')
    ax.plot(ages, trajectory_50, color='red', linewidth=2, label='50th Percentile')
    ax.plot(ages, trajectory_25, color='blue', linestyle='dashed', linewidth=2, label='25th Percentile')

    # 貯金ケース
    saving_trajectory = monthly_contribution * 12 * (ages - start_age)
    ax.plot(ages, saving_trajectory, color='green', linewidth=2, label='Saving Only')

    # 年齢と西暦を両方表示
    xtick_indices = list(range(0, len(ages), 5))  # 5年ごと
    xticks = ages[xtick_indices]
    xticklabels = [f"{age}\n({year})" for age, year in zip(ages[xtick_indices], years[xtick_indices])]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=10)
    
    # Y軸の上限を80パーセンタイルで設定
    y_max = np.percentile(final_values, 80)
    ax.set_ylim(0, y_max * 1.05)  # 少し余白
    
    ax.set_xlabel("Age(Year)")
    ax.set_ylabel("Amount (10,000 Yen)")
    ax.set_title("Investment Simulation")
    ax.legend()
    st.pyplot(fig)

    # ----------------------------
    # 🧾 結果数値の表示
    # ----------------------------
    st.markdown("### 💰 最終積立額（定年時）")
    st.metric("75パーセンタイル", f"{trajectory_75[-1]:,.0f} 万円")
    st.metric("50パーセンタイル（中央値）", f"{trajectory_50[-1]:,.0f} 万円")
    st.metric("25パーセンタイル", f"{trajectory_25[-1]:,.0f} 万円")
    st.metric("貯金のみの場合", f"{saving_trajectory[-1]:,.0f} 万円")

