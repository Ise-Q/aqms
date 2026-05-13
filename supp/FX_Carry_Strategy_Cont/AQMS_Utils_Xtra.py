import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import AQMS_Utils as aqms

def calc_contribution_to_IR(df, returns, cov_matrix):
    """
    Calculates the contribution of each asset to the portfolio's Information Ratio (IR).

    For each asset in the portfolio, the function measures how much the portfolio's 
    Information Ratio decreases when that asset is removed. The difference is interpreted 
    as the asset's contribution to the overall IR.

    Args:
        df (pd.DataFrame): DataFrame of portfolio holdings (weights), where columns represent assets.
        returns (pd.DataFrame): DataFrame of asset returns.
        cov_matrix (pd.DataFrame): Covariance matrix of asset returns.

    Returns:
        pd.DataFrame: A DataFrame indexed by asset names with a single column 
                      'contribution_to_IR' showing each asset's contribution to the portfolio IR.

    Notes:
        - The function assumes the existence of `calculate_portfolio_performance` and `target_vol` functions.
        - It rescales the holdings to a target volatility of 1% after dropping each asset.
    """
    
    
    df_out = pd.DataFrame(index = df.columns, columns=['contribution_to_IR'])
    
    results_full = aqms.calculate_portfolio_performance(df, returns, days_lag = 1)
    
    for i, x in enumerate(df.columns):
        holds_dropped = df.drop(x, axis=1).copy()
        holds_dropped, sf = aqms.target_vol(holds_dropped, cov_matrix, volatility = 0.01)
        results = aqms.calculate_portfolio_performance(holds_dropped, returns, days_lag = 1)
        df_out.loc[x] = results_full['perf_stats']['portfolio_info_ratio'][0] - results['perf_stats']['portfolio_info_ratio'][0]
        
    return df_out

def calc_lead_lag_IR(df, returns, lead_lag_range):
    """
    Calculates the portfolio's Information Ratio (IR) across different lead-lag shifts of the holdings.

    Shifts the holdings DataFrame by different lead-lag values and evaluates how the 
    Information Ratio changes for each shift. This can help assess the timing sensitivity 
    of the portfolio strategy.

    Args:
        df (pd.DataFrame): DataFrame of portfolio holdings (weights), where columns represent assets.
        returns (pd.DataFrame): DataFrame of asset returns.
        lead_lag_range (iterable): Range or list of integers representing the number of periods to shift holdings.
                                   Positive values imply leading (future) holdings, negative values imply lagging.

    Returns:
        pd.DataFrame: A DataFrame indexed by the lead-lag values, with a single column 
                      'portfolio_info_ratio' showing the IR for each shift.

    Notes:
        - The function assumes the existence of `calculate_portfolio_performance`.
        - Shifting introduces NaNs which are dropped (`dropna`) to maintain alignment.
        - `days_lag` parameter in performance calculation is set to 0 to match shifted holdings directly.
    """

    df_out = pd.DataFrame(index = lead_lag_range, columns=['portfolio_info_ratio'])
    for i, x in enumerate(lead_lag_range):
        holds_shifted = df.shift(x).dropna()
        results = aqms.calculate_portfolio_performance(holds_shifted, returns, days_lag = 0)
        df_out.iloc[i,:] = results['perf_stats']['portfolio_info_ratio']
        
    return df_out



def plot_bootstrapped_rtns(df_insample, df_all, num_bootstraps):
    """
    Plots bootstrapped cumulative return paths against actual out-of-sample returns.

    This function performs a bootstrap simulation on in-sample returns to generate 
    multiple hypothetical out-of-sample return paths. It compares these simulated paths 
    with the actual out-of-sample returns from the full return dataset.

    Parameters:
    ----------
    df_insample : pandas.DataFrame
        DataFrame of in-sample returns. Assumed to be a single-column DataFrame with a datetime index.
    
    df_all : pandas.DataFrame
        DataFrame of full returns including both in-sample and out-of-sample periods. 
        Must share the same structure and index format as df_insample.

    num_bootstraps : integer
        Number of bootstrap samples to use 
      

    Notes:
    ------
    - Generates 'num_bootstraps' bootstrapped return paths by randomly sampling with replacement 
      from in-sample returns.
    - Computes the cumulative return for each path and plots:
        - All bootstrapped paths
        - The mean of the bootstrapped paths
        - A 5–95% confidence band
        - The actual cumulative return in the out-of-sample period
    - Plots are aligned by date using the out-of-sample date index.

    Returns:
    -------
    None
        Displays a matplotlib plot.
    """    
    # -------------------
    # INPUTS
    # -------------------
    # Assume you have:
    # df_insample (your in-sample returns)
    # df_all (full returns, including in-sample and out-of-sample)
    
    n_paths = num_bootstraps
    n_days = len(df_all.index.difference(df_insample.index))  # OOS period length
    
    # -------------------
    # BOOTSTRAPPING
    # -------------------
    
    bootstrap_paths = []
    
    for i in range(n_paths):
        sampled_returns = df_insample.sample(n=n_days, replace=True).values.flatten()
        #cumulative_returns = (1 + sampled_returns).cumprod() - 1
        cumulative_returns = sampled_returns.cumsum()
        bootstrap_paths.append(cumulative_returns)
    
    bootstrap_paths = np.array(bootstrap_paths)  # Shape (n_paths, n_days)
    
    # -------------------
    # SUMMARY STATS
    # -------------------
    mean_path = np.mean(bootstrap_paths, axis=0)
    lower_band = np.percentile(bootstrap_paths, 5, axis=0)
    upper_band = np.percentile(bootstrap_paths, 95, axis=0)
    
    # Force them to pure floats
    mean_path = np.array(mean_path, dtype=float)
    lower_band = np.array(lower_band, dtype=float)
    upper_band = np.array(upper_band, dtype=float)
    
    # -------------------
    # OUT-OF-SAMPLE RETURNS
    # -------------------
    out_of_sample_dates = df_all.index.difference(df_insample.index)
    out_of_sample_returns = df_all.loc[out_of_sample_dates].sort_index()
    #out_of_sample_cum_returns = (1 + out_of_sample_returns).cumprod() - 1
    out_of_sample_cum_returns = out_of_sample_returns.cumsum()
    
    # -------------------
    # PLOTTING
    # -------------------
    
    x_dates = out_of_sample_returns.index
    x_numeric = np.arange(len(x_dates))
    
    plt.figure(figsize=(12, 7))
    
    # Plot all bootstrap paths
    for path in bootstrap_paths:
        plt.plot(x_numeric, path, color='lightblue', alpha=0.3)
    
    # Plot mean path
    plt.plot(x_numeric, mean_path, color='blue', lw=2, label='Mean Bootstrap Path')
    
    # Plot confidence interval
    plt.fill_between(
        x_numeric,
        lower_band,
        upper_band,
        color='blue',
        alpha=0.2,
        label='5-95% Confidence Band'
    )
    
    # Plot actual out-of-sample returns
    plt.plot(x_numeric, out_of_sample_cum_returns.values, color='black', lw=2.5, label='Actual Out-of-Sample')
    
    # Format x-axis
    ax = plt.gca()
    ax.set_xticks(x_numeric[::max(len(x_numeric)//10, 1)])
    ax.set_xticklabels([date.strftime('%Y-%m-%d') for date in x_dates[::max(len(x_numeric)//10, 1)]], rotation=45)
    
    plt.title('Bootstrapped Cumulative Returns vs Actual (Out-of-Sample)', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_cumsum_with_ir_bounds(daily_returns, periods_per_year=252, target_ir=0.5):
    """
    Plots cumulative returns, IR=0.5 mean line, and ±1 SD / ±2 SD envelopes.
    
    Args:
        daily_returns (pd.Series or pd.DataFrame): Daily returns (assumes simple returns).
        periods_per_year (int): Number of trading periods per year. Default is 252.
        target_ir (float): Target information ratio (default 0.5).
    """
    if isinstance(daily_returns, pd.DataFrame):
        # If DataFrame, assume first column
        daily_returns = daily_returns.iloc[:, 0]
    
    # Cumulative returns
    cumulative_returns = daily_returns.cumsum()
    
    # Time array
    t = np.arange(1, len(daily_returns) + 1)
    
    # Build expected mean return line (IR * sqrt(t) * daily volatility scaling)
    daily_vol = daily_returns.std()
    
    # The IR line increases linearly with time
    mean_line = pd.Series((target_ir * daily_vol * np.sqrt(periods_per_year)) * t / periods_per_year,
                          index=daily_returns.index)
    
    # Standard deviation grows with sqrt(time)
    std_dev_1 = pd.Series(daily_vol * np.sqrt(t), index=daily_returns.index)
    std_dev_2 = 2 * std_dev_1
    
    # Upper and lower bounds (±1 SD, ±2 SD)
    upper_1sd = mean_line + std_dev_1
    lower_1sd = mean_line - std_dev_1
    upper_2sd = mean_line + std_dev_2
    lower_2sd = mean_line - std_dev_2

    # Plot
    plt.figure(figsize=(14, 8))
    plt.plot(cumulative_returns, label='Cumulative Returns', color='blue')
    plt.plot(mean_line, label=f'Mean IR = {target_ir}', color='black', linestyle='--')
    plt.plot(upper_1sd, label='+1 SD', color='red', linestyle='dotted')
    plt.plot(lower_1sd, label='-1 SD', color='red', linestyle='dotted')
    plt.plot(upper_2sd, label='+2 SD', color='orange', linestyle='dotted')
    plt.plot(lower_2sd, label='-2 SD', color='orange', linestyle='dotted')
    
    plt.fill_between(daily_returns.index, lower_1sd, upper_1sd, color='red', alpha=0.1)
    plt.fill_between(daily_returns.index, lower_2sd, upper_2sd, color='orange', alpha=0.05)
    
    plt.title('Cumulative Returns with IR=0.5 Mean Line and Envelopes')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()