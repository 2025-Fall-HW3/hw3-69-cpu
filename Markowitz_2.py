import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import quantstats as qs
import gurobipy as gp 
import warnings
import argparse
import sys

# ... (omitted Project Setup and Data Initialization for brevity)
warnings.simplefilter(action="ignore", category=FutureWarning)

assets = [
    "SPY",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
]

# Initialize Bdf and df
Bdf = pd.DataFrame()
for asset in assets:
    raw = yf.download(asset, start="2012-01-01", end="2024-04-01", auto_adjust = False)
    Bdf[asset] = raw['Adj Close']

df = Bdf.loc["2019-01-01":"2024-04-01"]

"""
Strategy Creation

Create your own strategy, you can add parameter but please remain "price" and "exclude" unchanged
"""


class MyPortfolio:
    
    def __init__(self, price, exclude, 
                 mom_lookback=90, filter_lookback=25, top_k=2, # Main Strategy (M)
                 mom_lookback_short=20, filter_lookback_short=15, top_k_short=2, # Bootstrap Strategy (B)
                 mom_lookback_ultra_short=5, filter_lookback_ultra_short=5, top_k_ultra_short=4): # Ultra-Short Strategy (C)
        
        self.price = price
        self.returns = price.pct_change().fillna(0)
        self.exclude = exclude
        
        # Strategy M Parameters (User's best P4.1 parameters)
        self.mom_lookback = mom_lookback       
        self.filter_lookback = filter_lookback 
        self.top_k = top_k
        
        # Strategy B Parameters 
        self.mom_lookback_short = mom_lookback_short
        self.filter_lookback_short = filter_lookback_short
        self.top_k_short = top_k_short
        
        # Strategy C Parameters (New Ultra-Short Parameters)
        self.mom_lookback_ultra_short = mom_lookback_ultra_short
        self.filter_lookback_ultra_short = filter_lookback_ultra_short
        self.top_k_ultra_short = top_k_ultra_short


    def calculate_weights(self):
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_weights = pd.DataFrame(
            index=self.price.index, columns=self.price.columns
        ).fillna(0.0)

        # 計算三個策略的起始點
        start_main = max(self.mom_lookback, self.filter_lookback)
        start_bootstrap = max(self.mom_lookback_short, self.filter_lookback_short)
        start_ultra = max(self.mom_lookback_ultra_short, self.filter_lookback_ultra_short)
        
        # --- 0. 策略 C (Ultra-Short Regime) ---
        # 運行區間: i 從 start_ultra 到 start_bootstrap - 1
        for i in range(start_ultra, start_bootstrap):
            current_date = self.price.index[i]
            
            # 使用 Strategy C 的極短期參數
            mom_lb = self.mom_lookback_ultra_short
            filter_lb = self.filter_lookback_ultra_short
            top_k_val = self.top_k_ultra_short
            
            current_returns = self.returns.copy()[assets].iloc[:i]
            
            # 計算 Momentum and Filter
            momentum = (1 + current_returns.iloc[-mom_lb:]).prod(axis=0) - 1 
            filter_return = (1 + current_returns.iloc[-filter_lb:]).prod(axis=0) - 1 
            
            is_trending = (filter_return > 0)
            eligible_momentum = momentum[is_trending].sort_values(ascending=False)
            selected_assets = eligible_momentum.head(top_k_val).index
            
            # 分配 EQW 權重
            num_selected = len(selected_assets)
            if num_selected > 0:
                equal_weight = 1.0 / num_selected
                weights = pd.Series(0.0, index=assets)
                weights.loc[selected_assets] = equal_weight
            else:
                weights = pd.Series(0.0, index=assets)

            self.portfolio_weights.loc[current_date, assets] = weights.values


        # 1. 策略 B (Bootstrap/Warm-up Regime)
        # 運行區間: i 從 start_bootstrap 到 start_main - 1
        for i in range(start_bootstrap, start_main):
            current_date = self.price.index[i]
            
            # --- 使用 Strategy B 的短週期參數 ---
            mom_lb = self.mom_lookback_short
            filter_lb = self.filter_lookback_short
            top_k_val = self.top_k_short
            
            current_returns = self.returns.copy()[assets].iloc[:i]
            
            # 計算 Momentum and Filter
            momentum = (1 + current_returns.iloc[-mom_lb:]).prod(axis=0) - 1 
            filter_return = (1 + current_returns.iloc[-filter_lb:]).prod(axis=0) - 1 
            
            is_trending = (filter_return > 0)
            eligible_momentum = momentum[is_trending].sort_values(ascending=False)
            selected_assets = eligible_momentum.head(top_k_val).index
            
            # 分配 EQW 權重
            num_selected = len(selected_assets)
            if num_selected > 0:
                equal_weight = 1.0 / num_selected
                weights = pd.Series(0.0, index=assets)
                weights.loc[selected_assets] = equal_weight
            else:
                weights = pd.Series(0.0, index=assets)

            self.portfolio_weights.loc[current_date, assets] = weights.values

        # 2. 策略 M (Main Regime)
        # 運行區間: i 從 start_main 到最後
        for i in range(start_main, len(self.price)):
            current_date = self.price.index[i]
            
            # --- 使用 Strategy M 的主策略參數 ---
            mom_lb = self.mom_lookback
            filter_lb = self.filter_lookback
            top_k_val = self.top_k
            
            current_returns = self.returns.copy()[assets].iloc[:i]
            
            # 計算 Momentum and Filter
            mom_period_returns = current_returns.iloc[-mom_lb:]
            momentum = (1 + mom_period_returns).prod(axis=0) - 1 
            
            filter_period_returns = current_returns.iloc[-filter_lb:]
            filter_return = (1 + filter_period_returns).prod(axis=0) - 1 
            
            is_trending = (filter_return > 0)
            
            eligible_momentum = momentum[is_trending].sort_values(ascending=False)
            selected_assets = eligible_momentum.head(top_k_val).index
            
            # 分配 EQW 權重
            num_selected = len(selected_assets)
            if num_selected > 0:
                equal_weight = 1.0 / num_selected
                weights = pd.Series(0.0, index=assets)
                weights.loc[selected_assets] = equal_weight
            else:
                weights = pd.Series(0.0, index=assets)

            self.portfolio_weights.loc[current_date, assets] = weights.values


    # Note: calculate_portfolio_returns 和 get_results 保持不變
    def calculate_portfolio_returns(self):
        # 確保權重已計算
        if not hasattr(self, "portfolio_weights"):
            self.calculate_weights()

        # 計算投資組合報酬
        self.portfolio_returns = self.returns.copy()
        assets = self.price.columns[self.price.columns != self.exclude]
        self.portfolio_returns["Portfolio"] = (
            self.portfolio_returns[assets]
            .mul(self.portfolio_weights[assets])
            .sum(axis=1)
        )

    def get_results(self):
        # 確保投資組合報酬已計算
        if not hasattr(self, "portfolio_returns"):
            self.calculate_portfolio_returns()

        return self.portfolio_weights, self.portfolio_returns


if __name__ == "__main__":
    # Import grading system (protected file in GitHub Classroom)
    from grader_2 import AssignmentJudge
    
    parser = argparse.ArgumentParser(
        description="Introduction to Fintech Assignment 3 Part 12"
    )

    parser.add_argument(
        "--score",
        action="append",
        help="Score for assignment",
    )

    parser.add_argument(
        "--allocation",
        action="append",
        help="Allocation for asset",
    )

    parser.add_argument(
        "--performance",
        action="append",
        help="Performance for portfolio",
    )

    parser.add_argument(
        "--report", action="append", help="Report for evaluation metric"
    )

    parser.add_argument(
        "--cumulative", action="append", help="Cumulative product result"
    )

    args = parser.parse_args()

    judge = AssignmentJudge()
    
    # All grading logic is protected in grader_2.py
    judge.run_grading(args)