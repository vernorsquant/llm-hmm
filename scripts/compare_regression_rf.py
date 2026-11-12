"""Compare BSM vs FinBERT sentiment against SPY returns using Linear Regression and Random Forest.
Generates 4 distinct plots (and a 2x2 grid) comparing:
  1. Linear Regression: BSM score -> SPY return
  2. Linear Regression: FinBERT score -> SPY return
  3. Random Forest: BSM score -> SPY return
  4. Random Forest: FinBERT score -> SPY return
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from hmm_sentiment.market_data import fetch_market_data
from hmm_sentiment.config import FIG_DIR, TABLE_DIR

# Set style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 10

def main():
    # 1. Load sentiment series and market data
    series_path = os.path.join(TABLE_DIR, "labeler_daily_series.csv")
    sent_df = pd.read_csv(series_path)
    sent_df["Date"] = pd.to_datetime(sent_df["Date"])
    
    market_df = fetch_market_data().reset_index()
    market_df["Date"] = pd.to_datetime(market_df["Date"])
    
    # Merge on Date
    df = pd.merge(sent_df, market_df, on="Date").sort_values("Date").reset_index(drop=True)
    
    # Define targets:
    # Next-day return (causal trading: sentiment at close t -> return held over t+1)
    df["SPY_ret_next"] = df["SPY_ret"].shift(-1)
    
    # Drop last row with NaN next-day return
    valid_df = df.dropna(subset=["SPY_ret_next", "labeler_sent_w", "finbert_sent_w"]).copy()
    
    X_bsm = valid_df[["labeler_sent_w"]].values
    X_fin = valid_df[["finbert_sent_w"]].values
    y = valid_df["SPY_ret_next"].values * 100.0  # express return in percentage (%)
    
    print(f"Dataset shape: {len(valid_df)} trading days in 2023")
    
    # Grid for smooth RF and OLS curves
    bsm_grid = np.linspace(X_bsm.min() - 0.05, X_bsm.max() + 0.05, 300).reshape(-1, 1)
    fin_grid = np.linspace(X_fin.min() - 0.05, X_fin.max() + 0.05, 300).reshape(-1, 1)
    
    # --- Model 1: Linear Regression (BSM) ---
    ols_bsm = stats.linregress(X_bsm.flatten(), y)
    y_pred_ols_bsm = ols_bsm.intercept + ols_bsm.slope * bsm_grid.flatten()
    r2_ols_bsm = ols_bsm.rvalue ** 2
    mse_ols_bsm = np.mean((y - (ols_bsm.intercept + ols_bsm.slope * X_bsm.flatten())) ** 2)
    
    # --- Model 2: Linear Regression (FinBERT) ---
    ols_fin = stats.linregress(X_fin.flatten(), y)
    y_pred_ols_fin = ols_fin.intercept + ols_fin.slope * fin_grid.flatten()
    r2_ols_fin = ols_fin.rvalue ** 2
    mse_ols_fin = np.mean((y - (ols_fin.intercept + ols_fin.slope * X_fin.flatten())) ** 2)
    
    # --- Model 3: Random Forest (BSM) ---
    # Regularized to prevent trivial 1D overfitting, 5-fold CV
    rf_bsm = RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=10, random_state=42)
    rf_bsm.fit(X_bsm, y)
    y_pred_rf_bsm = rf_bsm.predict(bsm_grid)
    cv_r2_bsm = np.mean(cross_val_score(rf_bsm, X_bsm, y, cv=KFold(5, shuffle=True, random_state=42)))
    mse_rf_bsm = mean_squared_error(y, rf_bsm.predict(X_bsm))
    
    # --- Model 4: Random Forest (FinBERT) ---
    rf_fin = RandomForestRegressor(n_estimators=200, max_depth=3, min_samples_leaf=10, random_state=42)
    rf_fin.fit(X_fin, y)
    y_pred_rf_fin = rf_fin.predict(fin_grid)
    cv_r2_fin = np.mean(cross_val_score(rf_fin, X_fin, y, cv=KFold(5, shuffle=True, random_state=42)))
    mse_rf_fin = mean_squared_error(y, rf_fin.predict(X_fin))
    
    # Print metrics table
    summary_data = [
        {"Model": "Linear Regression", "Feature": "BSM (LLM-Labeler)", "Slope (β)": f"{ols_bsm.slope:.4f}", "Intercept (α)": f"{ols_bsm.intercept:.4f}", "R²": f"{r2_ols_bsm:.4f}", "p-value": f"{ols_bsm.pvalue:.4f}", "MSE (%²)": f"{mse_ols_bsm:.4f}"},
        {"Model": "Linear Regression", "Feature": "FinBERT Sentiment", "Slope (β)": f"{ols_fin.slope:.4f}", "Intercept (α)": f"{ols_fin.intercept:.4f}", "R²": f"{r2_ols_fin:.4f}", "p-value": f"{ols_fin.pvalue:.4f}", "MSE (%²)": f"{mse_ols_fin:.4f}"},
        {"Model": "Random Forest", "Feature": "BSM (LLM-Labeler)", "Slope (β)": "N/A (Non-linear)", "Intercept (α)": "N/A", "R² (5-fold CV)": f"{cv_r2_bsm:.4f}", "p-value": "N/A", "MSE (%²)": f"{mse_rf_bsm:.4f}"},
        {"Model": "Random Forest", "Feature": "FinBERT Sentiment", "Slope (β)": "N/A (Non-linear)", "Intercept (α)": "N/A", "R² (5-fold CV)": f"{cv_r2_fin:.4f}", "p-value": "N/A", "MSE (%²)": f"{mse_rf_fin:.4f}"},
    ]
    summary_df = pd.DataFrame(summary_data)
    print("\n=== Model Performance Summary ===")
    print(summary_df.to_string(index=False))
    
    summary_csv_path = os.path.join(TABLE_DIR, "regression_rf_comparison_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary table to {summary_csv_path}")
    
    # Colors
    c_blue = "#2563EB"
    c_red = "#DC2626"
    c_green = "#059669"
    c_purple = "#7C3AED"
    c_dot = "#475569"
    
    # ---------------- 2x2 Combined Figure ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharey=True)
    
    # Plot 1: OLS BSM
    ax1 = axes[0, 0]
    ax1.scatter(X_bsm, y, color=c_dot, alpha=0.55, edgecolors="none", s=28, label="Daily observations")
    ax1.plot(bsm_grid, y_pred_ols_bsm, color=c_blue, lw=2.4, label=f"OLS fit: y = {ols_bsm.slope:+.3f}x + {ols_bsm.intercept:+.3f}")
    ax1.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax1.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax1.set_title("Plot 1: Linear Regression (BSM Score -> SPY Return)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("BSM Impact Score ($t$)")
    ax1.set_ylabel("Next-Day SPY Return ($t+1$, %)")
    ax1.legend(loc="upper left", frameon=True, fontsize=9)
    ax1.text(0.04, 0.06, f"$R^2 = {r2_ols_bsm:.4f}$\n$p = {ols_bsm.pvalue:.4f}$\nMSE = {mse_ols_bsm:.4f}", transform=ax1.transAxes,
             fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="#EFF6FF", ec=c_blue, alpha=0.9))
    
    # Plot 2: OLS FinBERT
    ax2 = axes[0, 1]
    ax2.scatter(X_fin, y, color=c_dot, alpha=0.55, edgecolors="none", s=28, label="Daily observations")
    ax2.plot(fin_grid, y_pred_ols_fin, color=c_red, lw=2.4, label=f"OLS fit: y = {ols_fin.slope:+.3f}x + {ols_fin.intercept:+.3f}")
    ax2.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax2.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax2.set_title("Plot 2: Linear Regression (FinBERT Sentiment -> SPY Return)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("FinBERT Sentiment Score ($t$)")
    ax2.legend(loc="upper left", frameon=True, fontsize=9)
    ax2.text(0.04, 0.06, f"$R^2 = {r2_ols_fin:.4f}$\n$p = {ols_fin.pvalue:.4f}$\nMSE = {mse_ols_fin:.4f}", transform=ax2.transAxes,
             fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="#FEF2F2", ec=c_red, alpha=0.9))
    
    # Plot 3: RF BSM
    ax3 = axes[1, 0]
    ax3.scatter(X_bsm, y, color=c_dot, alpha=0.55, edgecolors="none", s=28, label="Daily observations")
    ax3.plot(bsm_grid, y_pred_rf_bsm, color=c_green, lw=2.4, label="Random Forest response curve")
    ax3.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax3.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax3.set_title("Plot 3: Random Forest (BSM Score -> SPY Return)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("BSM Impact Score ($t$)")
    ax3.set_ylabel("Next-Day SPY Return ($t+1$, %)")
    ax3.legend(loc="upper left", frameon=True, fontsize=9)
    ax3.text(0.04, 0.06, f"CV $R^2 = {cv_r2_bsm:.4f}$\nMSE = {mse_rf_bsm:.4f}\nTrees = 200, Depth = 3", transform=ax3.transAxes,
             fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec=c_green, alpha=0.9))
    
    # Plot 4: RF FinBERT
    ax4 = axes[1, 1]
    ax4.scatter(X_fin, y, color=c_dot, alpha=0.55, edgecolors="none", s=28, label="Daily observations")
    ax4.plot(fin_grid, y_pred_rf_fin, color=c_purple, lw=2.4, label="Random Forest response curve")
    ax4.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax4.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax4.set_title("Plot 4: Random Forest (FinBERT Sentiment -> SPY Return)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("FinBERT Sentiment Score ($t$)")
    ax4.legend(loc="upper left", frameon=True, fontsize=9)
    ax4.text(0.04, 0.06, f"CV $R^2 = {cv_r2_fin:.4f}$\nMSE = {mse_rf_fin:.4f}\nTrees = 200, Depth = 3", transform=ax4.transAxes,
             fontsize=9.5, bbox=dict(boxstyle="round,pad=0.4", fc="#FAF5FF", ec=c_purple, alpha=0.9))
    
    plt.suptitle("Predictive Relationship: BSM Impact & FinBERT Sentiment vs. SPY Returns (2023)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    grid_path = os.path.join(FIG_DIR, "sentiment_vs_returns_regression_rf.png")
    plt.savefig(grid_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved 2x2 grid plot to {grid_path}")
    
    # ---------------- 4 Individual Plots ----------------
    # Plot 1 standalone
    fig1, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_bsm, y, color=c_dot, alpha=0.55, s=32, label="Daily observations")
    ax.plot(bsm_grid, y_pred_ols_bsm, color=c_blue, lw=2.5, label=f"OLS: y = {ols_bsm.slope:+.3f}x + {ols_bsm.intercept:+.3f}")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_title("Linear Regression: BSM Score vs. Next-Day SPY Return", fontweight="bold")
    ax.set_xlabel("BSM Impact Score ($t$)")
    ax.set_ylabel("Next-Day SPY Return ($t+1$, %)")
    ax.legend(frameon=True)
    ax.text(0.05, 0.06, f"$R^2 = {r2_ols_bsm:.4f}$\n$p$-value = {ols_bsm.pvalue:.4f}\nMSE = {mse_ols_bsm:.4f}",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.4", fc="#EFF6FF", ec=c_blue, alpha=0.9))
    plt.tight_layout()
    p1 = os.path.join(FIG_DIR, "plot1_linear_regression_bsm.png")
    fig1.savefig(p1, dpi=200)
    plt.close(fig1)

    # Plot 2 standalone
    fig2, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_fin, y, color=c_dot, alpha=0.55, s=32, label="Daily observations")
    ax.plot(fin_grid, y_pred_ols_fin, color=c_red, lw=2.5, label=f"OLS: y = {ols_fin.slope:+.3f}x + {ols_fin.intercept:+.3f}")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_title("Linear Regression: FinBERT Sentiment vs. Next-Day SPY Return", fontweight="bold")
    ax.set_xlabel("FinBERT Sentiment Score ($t$)")
    ax.set_ylabel("Next-Day SPY Return ($t+1$, %)")
    ax.legend(frameon=True)
    ax.text(0.05, 0.06, f"$R^2 = {r2_ols_fin:.4f}$\n$p$-value = {ols_fin.pvalue:.4f}\nMSE = {mse_ols_fin:.4f}",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.4", fc="#FEF2F2", ec=c_red, alpha=0.9))
    plt.tight_layout()
    p2 = os.path.join(FIG_DIR, "plot2_linear_regression_finbert.png")
    fig2.savefig(p2, dpi=200)
    plt.close(fig2)

    # Plot 3 standalone
    fig3, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_bsm, y, color=c_dot, alpha=0.55, s=32, label="Daily observations")
    ax.plot(bsm_grid, y_pred_rf_bsm, color=c_green, lw=2.5, label="Random Forest curve")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_title("Random Forest: BSM Score vs. Next-Day SPY Return", fontweight="bold")
    ax.set_xlabel("BSM Impact Score ($t$)")
    ax.set_ylabel("Next-Day SPY Return ($t+1$, %)")
    ax.legend(frameon=True)
    ax.text(0.05, 0.06, f"5-Fold CV $R^2 = {cv_r2_bsm:.4f}$\nMSE = {mse_rf_bsm:.4f}\nTrees = 200, Max Depth = 3",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.4", fc="#ECFDF5", ec=c_green, alpha=0.9))
    plt.tight_layout()
    p3 = os.path.join(FIG_DIR, "plot3_random_forest_bsm.png")
    fig3.savefig(p3, dpi=200)
    plt.close(fig3)

    # Plot 4 standalone
    fig4, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_fin, y, color=c_dot, alpha=0.55, s=32, label="Daily observations")
    ax.plot(fin_grid, y_pred_rf_fin, color=c_purple, lw=2.5, label="Random Forest curve")
    ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.axvline(0, color="gray", ls="--", lw=0.8, alpha=0.7)
    ax.set_title("Random Forest: FinBERT Sentiment vs. Next-Day SPY Return", fontweight="bold")
    ax.set_xlabel("FinBERT Sentiment Score ($t$)")
    ax.set_ylabel("Next-Day SPY Return ($t+1$, %)")
    ax.legend(frameon=True)
    ax.text(0.05, 0.06, f"5-Fold CV $R^2 = {cv_r2_fin:.4f}$\nMSE = {mse_rf_fin:.4f}\nTrees = 200, Max Depth = 3",
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.4", fc="#FAF5FF", ec=c_purple, alpha=0.9))
    plt.tight_layout()
    p4 = os.path.join(FIG_DIR, "plot4_random_forest_finbert.png")
    fig4.savefig(p4, dpi=200)
    plt.close(fig4)

    print("All individual plots successfully generated.")

if __name__ == "__main__":
    main()
