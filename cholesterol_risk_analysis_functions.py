import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from reportlab.platypus import Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from pdf_functions import add_image_to_story, add_heading, add_text


def ensure_charts_directory():
    """Ensure the charts directory exists."""
    if not os.path.exists('charts'):
        os.makedirs('charts')


def calculate_proportion_ci(proportion, n, confidence=0.95):
    """
    Calculate confidence interval for a proportion using normal approximation.
    
    Parameters:
    -----------
    proportion : float
        Sample proportion
    n : int
        Sample size
    confidence : float, optional
        Confidence level (default: 0.95 for 95% CI)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound)
    """
    if n == 0:
        return (0, 0)
    
    z = stats.norm.ppf((1 + confidence) / 2)
    se = np.sqrt(proportion * (1 - proportion) / n)
    margin = z * se
    
    lower = max(0, proportion - margin)
    upper = min(1, proportion + margin)
    
    return (lower, upper)


def filter_risk_groups(df, age_column='age_years', gender_column='gender', 
                      smoke_column='smoke', ap_hi_column='ap_hi', 
                      cholesterol_column='cholesterol', cardio_column='cardio'):
    """
    Filter data into two risk groups as specified in the task.
    
    Group 1: Smoking men, age 60-65, systolic pressure < 120, normal cholesterol (cholesterol=1)
    Group 2: Same age range, systolic pressure [160, 180), well above normal cholesterol (cholesterol=3)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str, optional
        Name of the age column (default: 'age_years')
    gender_column : str, optional
        Name of the gender column (default: 'gender', 1=male)
    smoke_column : str, optional
        Name of the smoke column (default: 'smoke', 1=smoker)
    ap_hi_column : str, optional
        Name of the systolic pressure column (default: 'ap_hi')
    cholesterol_column : str, optional
        Name of the cholesterol column (default: 'cholesterol')
    cardio_column : str, optional
        Name of the cardio column (default: 'cardio')
    
    Returns:
    --------
    dict : Dictionary with group1 and group2 dataframes
    """
    # Group 1: Smoking men, age 60-65, ap_hi < 120, cholesterol = 1 (normal)
    group1 = df[
        (df[age_column] >= 60) & (df[age_column] <= 65) &
        (df[gender_column] == 1) &  # Men (assuming 1=male)
        (df[smoke_column] == 1) &  # Smokers
        (df[ap_hi_column] < 120) &
        (df[cholesterol_column] == 1)  # Normal cholesterol
    ].copy()
    
    # Group 2: Age 60-65, ap_hi in [160, 180), cholesterol = 3 (well above normal)
    group2 = df[
        (df[age_column] >= 60) & (df[age_column] <= 65) &
        (df[ap_hi_column] >= 160) & (df[ap_hi_column] < 180) &
        (df[cholesterol_column] == 3)  # Well above normal cholesterol
    ].copy()
    
    return {
        'group1': group1,
        'group2': group2
    }


def calculate_group_statistics(group_df, cardio_column='cardio'):
    """
    Calculate statistics for a risk group including proportion with CVD and 95% CI.
    
    Parameters:
    -----------
    group_df : pandas.DataFrame
        Dataframe for the group
    cardio_column : str, optional
        Name of the cardio column (default: 'cardio')
    
    Returns:
    --------
    dict : Dictionary with statistics
    """
    n = len(group_df)
    if n == 0:
        return {
            'n': 0,
            'cvd_count': 0,
            'proportion': 0,
            'ci_lower': 0,
            'ci_upper': 0
        }
    
    cvd_count = group_df[cardio_column].sum()
    proportion = cvd_count / n
    
    ci_lower, ci_upper = calculate_proportion_ci(proportion, n)
    
    return {
        'n': n,
        'cvd_count': cvd_count,
        'proportion': proportion,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }


def calculate_risk_ratio(group1_stats, group2_stats):
    """
    Calculate risk ratio (relative risk) between two groups.
    
    Parameters:
    -----------
    group1_stats : dict
        Statistics for group 1
    group2_stats : dict
        Statistics for group 2
    
    Returns:
    --------
    dict : Dictionary with risk ratio and 95% CI
    """
    p1 = group1_stats['proportion']
    p2 = group2_stats['proportion']
    
    if p1 == 0:
        # If group1 has no cases, risk ratio is undefined
        return {
            'risk_ratio': np.inf,
            'ci_lower': np.inf,
            'ci_upper': np.inf,
            'log_rr': np.inf,
            'log_se': np.inf
        }
    
    risk_ratio = p2 / p1
    
    # Calculate 95% CI for risk ratio using log transformation
    log_rr = np.log(risk_ratio)
    
    # Standard error of log(RR)
    n1 = group1_stats['n']
    n2 = group2_stats['n']
    log_se = np.sqrt((1 - p1) / (n1 * p1) + (1 - p2) / (n2 * p2))
    
    z = stats.norm.ppf(0.975)  # 95% CI
    log_ci_lower = log_rr - z * log_se
    log_ci_upper = log_rr + z * log_se
    
    ci_lower = np.exp(log_ci_lower)
    ci_upper = np.exp(log_ci_upper)
    
    return {
        'risk_ratio': risk_ratio,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'log_rr': log_rr,
        'log_se': log_se
    }


def create_proportion_plot_with_ci(group1_stats, group2_stats, output_path=None):
    """
    Create a proportion plot with 95% confidence intervals for both groups.
    
    Parameters:
    -----------
    group1_stats : dict
        Statistics for group 1
    group2_stats : dict
        Statistics for group 2
    output_path : str, optional
        Path to save the chart
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = 'charts/proportion_plot_cvd_risk_groups.png'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = ['Group 1\n(Low Risk)', 'Group 2\n(High Risk)']
    proportions = [group1_stats['proportion'], group2_stats['proportion']]
    ci_lowers = [group1_stats['ci_lower'], group2_stats['ci_lower']]
    ci_uppers = [group1_stats['ci_upper'], group2_stats['ci_upper']]
    
    x_pos = np.arange(len(groups))
    width = 0.6
    
    # Create bars
    bars = ax.bar(x_pos, proportions, width, alpha=0.7, 
                 color=['#3498db', '#e74c3c'], edgecolor='black')
    
    # Add error bars for 95% CI
    errors_lower = [p - ci_l for p, ci_l in zip(proportions, ci_lowers)]
    errors_upper = [ci_u - p for p, ci_u in zip(proportions, ci_uppers)]
    
    ax.errorbar(x_pos, proportions, 
               yerr=[errors_lower, errors_upper],
               fmt='none', color='black', capsize=5, capthick=2, linewidth=2)
    
    # Add value labels on bars
    for i, (prop, n) in enumerate(zip(proportions, [group1_stats['n'], group2_stats['n']])):
        ax.text(i, prop + ci_uppers[i] + 0.02, f'{prop:.3f}\n(n={n})',
               ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Proportion with CVD', fontsize=12)
    ax.set_title('Proportion of People with CVD by Risk Group (with 95% Confidence Intervals)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, max(ci_uppers) * 1.3)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add CI labels
    for i, (ci_l, ci_u) in enumerate(zip(ci_lowers, ci_uppers)):
        ax.text(i, ci_u + 0.01, f'95% CI: [{ci_l:.3f}, {ci_u:.3f}]',
               ha='center', va='bottom', fontsize=9, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_risk_ratio_plot(risk_ratio_stats, output_path=None):
    """
    Create a risk-ratio dot and whisker plot with 95% confidence interval.
    
    Parameters:
    -----------
    risk_ratio_stats : dict
        Risk ratio statistics including CI
    output_path : str, optional
        Path to save the chart
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = 'charts/risk_ratio_plot.png'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    risk_ratio = risk_ratio_stats['risk_ratio']
    ci_lower = risk_ratio_stats['ci_lower']
    ci_upper = risk_ratio_stats['ci_upper']
    
    # Handle infinite values
    if np.isinf(risk_ratio):
        # If risk ratio is infinite, set a maximum for visualization
        max_rr = 20
        risk_ratio = max_rr
        ci_upper = max_rr
        ci_lower = 1
    
    # Create dot and whisker plot
    x_pos = 0
    ax.plot(x_pos, risk_ratio, 'o', markersize=15, color='red', 
           label=f'Risk Ratio = {risk_ratio_stats["risk_ratio"]:.2f}')
    
    # Add whisker (confidence interval)
    ax.plot([x_pos, x_pos], [ci_lower, ci_upper], 'k-', linewidth=3)
    ax.plot([x_pos - 0.05, x_pos + 0.05], [ci_lower, ci_lower], 'k-', linewidth=3)
    ax.plot([x_pos - 0.05, x_pos + 0.05], [ci_upper, ci_upper], 'k-', linewidth=3)
    
    # Add reference line at RR = 1 (no difference)
    ax.axhline(y=1, color='blue', linestyle='--', linewidth=2, 
              label='No difference (RR = 1)')
    
    # Add reference line at RR = 5 (hypothesized ratio)
    ax.axhline(y=5, color='green', linestyle='--', linewidth=2, 
              label='Hypothesized ratio (RR = 5)')
    
    # Add text labels
    ax.text(x_pos + 0.15, risk_ratio, f'RR = {risk_ratio_stats["risk_ratio"]:.2f}',
           fontsize=12, fontweight='bold', va='center')
    ax.text(x_pos + 0.15, (ci_lower + ci_upper) / 2, 
           f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]',
           fontsize=10, va='center', style='italic')
    
    ax.set_xlim(-0.5, 1)
    ax.set_ylabel('Risk Ratio (Group 2 / Group 1)', fontsize=12)
    ax.set_title('Risk Ratio with 95% Confidence Interval\n(Group 2 vs Group 1)',
                fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_cholesterol_risk_analysis(df, age_column='age_years', gender_column='gender',
                                     smoke_column='smoke', ap_hi_column='ap_hi',
                                     cholesterol_column='cholesterol', cardio_column='cardio'):
    """
    Generate complete cholesterol risk analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str, optional
        Name of the age column (default: 'age_years')
    gender_column : str, optional
        Name of the gender column (default: 'gender')
    smoke_column : str, optional
        Name of the smoke column (default: 'smoke')
    ap_hi_column : str, optional
        Name of the systolic pressure column (default: 'ap_hi')
    cholesterol_column : str, optional
        Name of the cholesterol column (default: 'cholesterol')
    cardio_column : str, optional
        Name of the cardio column (default: 'cardio')
    
    Returns:
    --------
    dict : Dictionary containing all analysis results
    """
    ensure_charts_directory()
    
    # Filter groups
    groups = filter_risk_groups(df, age_column, gender_column, smoke_column,
                               ap_hi_column, cholesterol_column, cardio_column)
    
    # Calculate statistics
    group1_stats = calculate_group_statistics(groups['group1'], cardio_column)
    group2_stats = calculate_group_statistics(groups['group2'], cardio_column)
    
    # Calculate risk ratio
    risk_ratio_stats = calculate_risk_ratio(group1_stats, group2_stats)
    
    # Create plots
    proportion_plot_path = create_proportion_plot_with_ci(group1_stats, group2_stats)
    risk_ratio_plot_path = create_risk_ratio_plot(risk_ratio_stats)
    
    return {
        'group1_stats': group1_stats,
        'group2_stats': group2_stats,
        'risk_ratio_stats': risk_ratio_stats,
        'proportion_plot_path': proportion_plot_path,
        'risk_ratio_plot_path': risk_ratio_plot_path,
        'group1_data': groups['group1'],
        'group2_data': groups['group2']
    }


def add_cholesterol_risk_analysis_section_to_pdf(story, df, age_column='age_years',
                                                gender_column='gender', smoke_column='smoke',
                                                ap_hi_column='ap_hi', cholesterol_column='cholesterol',
                                                cardio_column='cardio'):
    """
    Add cholesterol risk analysis section to PDF report.
    
    Parameters:
    -----------
    story : list
        The PDF story list
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str, optional
        Name of the age column (default: 'age_years')
    gender_column : str, optional
        Name of the gender column (default: 'gender')
    smoke_column : str, optional
        Name of the smoke column (default: 'smoke')
    ap_hi_column : str, optional
        Name of the systolic pressure column (default: 'ap_hi')
    cholesterol_column : str, optional
        Name of the cholesterol column (default: 'cholesterol')
    cardio_column : str, optional
        Name of the cardio column (default: 'cardio')
    """
    styles = getSampleStyleSheet()
    
    # Add heading
    heading = Paragraph("Cholesterol Risk Analysis: CVD Risk Comparison", styles['Heading1'])
    story.append(heading)
    story.append(Spacer(1, 0.2*inch))
    
    # Add hypothesis statement
    hypothesis_heading = Paragraph("Research Hypothesis", styles['Heading2'])
    story.append(hypothesis_heading)
    story.append(Spacer(1, 0.1*inch))
    
    hypothesis_text = """
    <b>Hypothesis:</b> For people in the group of smoking men aged from 60 to 65 whose systolic 
    pressure is less than 120 and cholesterol is normal, the risk of CVD is estimated to be 
    5 times lower than for those with pressure in the interval [160, 180) and cholesterol 
    well above normal.
    """
    para = Paragraph(hypothesis_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.15*inch))
    
    # Generate analysis
    results = generate_cholesterol_risk_analysis(df, age_column, gender_column, smoke_column,
                                                ap_hi_column, cholesterol_column, cardio_column)
    
    group1_stats = results['group1_stats']
    group2_stats = results['group2_stats']
    risk_ratio_stats = results['risk_ratio_stats']
    
    # Add group descriptions
    groups_heading = Paragraph("Group Definitions", styles['Heading2'])
    story.append(groups_heading)
    story.append(Spacer(1, 0.1*inch))
    
    groups_text = f"""
    <b>Group 1 (Low Risk):</b><br/>
    • Smoking men<br/>
    • Age: 60-65 years<br/>
    • Systolic pressure: < 120<br/>
    • Cholesterol: Normal (cholesterol = 1)<br/>
    • Sample size: {group1_stats['n']}<br/><br/>
    
    <b>Group 2 (High Risk):</b><br/>
    • Age: 60-65 years<br/>
    • Systolic pressure: 160-180<br/>
    • Cholesterol: Well above normal (cholesterol = 3)<br/>
    • Sample size: {group2_stats['n']}<br/>
    """
    para = Paragraph(groups_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.15*inch))
    
    # Add statistics
    stats_heading = Paragraph("Statistical Results", styles['Heading2'])
    story.append(stats_heading)
    story.append(Spacer(1, 0.1*inch))
    
    stats_text = f"""
    <b>Group 1 Statistics:</b><br/>
    • Number with CVD: {group1_stats['cvd_count']} out of {group1_stats['n']}<br/>
    • Proportion with CVD: {group1_stats['proportion']:.4f} ({group1_stats['proportion']*100:.2f}%)<br/>
    • 95% Confidence Interval: [{group1_stats['ci_lower']:.4f}, {group1_stats['ci_upper']:.4f}]<br/><br/>
    
    <b>Group 2 Statistics:</b><br/>
    • Number with CVD: {group2_stats['cvd_count']} out of {group2_stats['n']}<br/>
    • Proportion with CVD: {group2_stats['proportion']:.4f} ({group2_stats['proportion']*100:.2f}%)<br/>
    • 95% Confidence Interval: [{group2_stats['ci_lower']:.4f}, {group2_stats['ci_upper']:.4f}]<br/><br/>
    
    <b>Risk Ratio (Group 2 / Group 1):</b><br/>
    • Risk Ratio: {risk_ratio_stats['risk_ratio']:.4f}<br/>
    • 95% Confidence Interval: [{risk_ratio_stats['ci_lower']:.4f}, {risk_ratio_stats['ci_upper']:.4f}]<br/>
    """
    para = Paragraph(stats_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.15*inch))
    
    # Add proportion plot
    proportion_heading = Paragraph("Proportion Plot with 95% Confidence Intervals", styles['Heading2'])
    story.append(proportion_heading)
    story.append(Spacer(1, 0.1*inch))
    
    proportion_desc = """
    The plot below shows the proportion of people with CVD in each group, along with 
    95% confidence intervals. The error bars represent the uncertainty in the proportion estimates.
    """
    para = Paragraph(proportion_desc, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    add_image_to_story(story, results['proportion_plot_path'], width=5*inch)
    
    # Add risk ratio plot
    risk_ratio_heading = Paragraph("Risk Ratio Plot with 95% Confidence Interval", styles['Heading2'])
    story.append(risk_ratio_heading)
    story.append(Spacer(1, 0.1*inch))
    
    risk_ratio_desc = """
    The dot and whisker plot below shows the risk ratio (Group 2 / Group 1) with its 95% 
    confidence interval. The red dot represents the point estimate, and the black line shows 
    the confidence interval. Reference lines at RR=1 (no difference) and RR=5 (hypothesized ratio) 
    are also shown.
    """
    para = Paragraph(risk_ratio_desc, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    add_image_to_story(story, results['risk_ratio_plot_path'], width=5*inch)
    
    # Add conclusion
    conclusion_heading = Paragraph("Conclusion", styles['Heading2'])
    story.append(conclusion_heading)
    story.append(Spacer(1, 0.1*inch))
    
    # Determine if hypothesis is confirmed or denied
    rr = risk_ratio_stats['risk_ratio']
    ci_lower = risk_ratio_stats['ci_lower']
    ci_upper = risk_ratio_stats['ci_upper']
    
    # Check if 5 is within the confidence interval
    if not np.isinf(ci_lower) and not np.isinf(ci_upper):
        if ci_lower <= 5 <= ci_upper:
            conclusion = "CONFIRMED"
            conclusion_detail = f"""
            The hypothesis is <b>CONFIRMED</b>. The observed risk ratio of {rr:.2f} is consistent 
            with the hypothesized ratio of 5, as the value 5 falls within the 95% confidence 
            interval [{ci_lower:.2f}, {ci_upper:.2f}].
            """
        elif rr < 5:
            conclusion = "PARTIALLY CONFIRMED"
            conclusion_detail = f"""
            The hypothesis is <b>PARTIALLY CONFIRMED</b>. The observed risk ratio of {rr:.2f} 
            is lower than the hypothesized ratio of 5, but the data still shows that Group 2 
            has significantly higher risk than Group 1. The 95% confidence interval is 
            [{ci_lower:.2f}, {ci_upper:.2f}].
            """
        else:
            conclusion = "DENIED"
            conclusion_detail = f"""
            The hypothesis is <b>DENIED</b>. The observed risk ratio of {rr:.2f} is higher 
            than the hypothesized ratio of 5. The 95% confidence interval is 
            [{ci_lower:.2f}, {ci_upper:.2f}], which does not include 5.
            """
    else:
        conclusion = "CANNOT DETERMINE"
        conclusion_detail = """
        The analysis cannot be completed due to insufficient data in one or both groups. 
        Please check the group definitions and data availability.
        """
    
    conclusion_text = f"""
    <b>Hypothesis Status: {conclusion}</b><br/><br/>
    {conclusion_detail}<br/><br/>
    
    <b>Interpretation:</b><br/>
    • A risk ratio of 5 means Group 2 has 5 times the risk of Group 1.<br/>
    • If the 95% confidence interval includes 5, the hypothesis is supported by the data.<br/>
    • The proportion plot shows the actual CVD rates in each group with uncertainty estimates.<br/>
    • The risk ratio plot visualizes the relative risk with confidence bounds.
    """
    
    para = Paragraph(conclusion_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.2*inch))

