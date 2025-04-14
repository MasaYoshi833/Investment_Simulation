# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 00:48:46 2025

@author: my199
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# タイトル
st.title("積立投資シミュレーション")

# -------------------------------
# 📌 前提条件を表示
# -------------------------------
st.markdown("### 📊 投資前提")
st.markdown("""
- 株式：リターン **5.5%**, リスク **23%**
- 債券：リターン **0.9%**, リスク **3%**
- 株と債券の相関係数：**-0.3**
""")

# -------------------------------
# 🔧 ユーザー入力
# -------------------------------
st.sidebar.header("シミュレーション設定")

start_age = st.sidebar.slider("現在の年齢", 20, 60, 30)
monthly_contribution = st.sidebar.number_input("毎月の投資額（万円）", min_value=1, max_value=30, value=5)
equity_ratio = st.sidebar.slider("株式の比率（％）", 0, 100, 50)
retirement_age = 65

# 実行ボタン
run_simulation = st.button("💡 シミュレーションを実行")

if run_simulation:
    # -------------------------------
    # 投資シミュレーション本体
    # -------------------------------

    # パラメータ設定
    equity_return = 0.055
    bond_return = 0.009
    inflation = 0.02
    equity_vol = 0.23
    bond_vol = 0.03
    correlation = -0.3

    returnYearly = np.array([equity_return, bond_return])
    volatilityYearly = np.array([equity_vol, bond_vol])
    corrYearly = np.array([[1, correlation], [correlation, 1]])

    # 月次変換
    monthly_returns = returnYearly / 12
    monthly_volatility = volatilityYearly / np.sqrt(12)
    cov_matrix = np.diag(monthly_volatility) @ corrYearly @ np.diag(monthly_volatility)

    # 投資条件
    weights = np.array([equity_ratio / 100, 1 - equity_ratio / 100])
    n_simulations = 1000
    n_years = retirement_age - start_age
    n_months = n_years * 12
    ages = np.arange(start_age, retirement_age + 1)

    # シミュレーション実行
    all_trajectories = np.zeros((n_simulations, n_years + 1))
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

    # パーセンタイル抽出
    final_values = all_trajectories[:, -1]
    idx_25 = np.abs(final_values - np.percentile(final_values, 25)).argmin()
    idx_50 = np.abs(final_values - np.percentile(final_values, 50)).argmin()
    idx_75 = np.abs(final_values - np.percentile(final_values, 75)).argmin()

    trajectory_25 = all_trajectories[idx_25]
    trajectory_50 = all_trajectories[idx_50]
    trajectory_75 = all_trajectories[idx_75]

    # -------------------------------
    # 📈 グラフ描画
    # -------------------------------
    st.subheader("📈 投資シミュレーション結果")

    fig, ax = plt.subplots(figsize=(12, 6))

    # 全シミュレーション（グレー）
    for i in range(n_simulations):
        ax.plot(ages, all_trajectories[i], color='gray', alpha=0.05)

    # 25%・50%・75%のシナリオ
    ax.plot(ages, trajectory_75, color='blue', linewidth=2, linestyle='dashed', label='75th Percentile')
    ax.plot(ages, trajectory_50, color='red', linewidth=2, label='Median (50th)')
    ax.plot(ages, trajectory_25, color='blue', linewidth=2, linestyle='dashed', label='25th Percentile')

    # 貯金のみのライン
    saving_trajectory = monthly_contribution * 12 * (ages - start_age)
    ax.plot(ages, saving_trajectory, color='green', linewidth=2, label='Saving Only')

    ax.set_title("積立投資シミュレーション（退職まで）")
    ax.set_xlabel("年齢")
    ax.set_ylabel("資産額（万円）")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
