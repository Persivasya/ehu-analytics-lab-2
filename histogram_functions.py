import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import Image, Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from pdf_functions import add_image_to_story


def ensure_charts_directory():
    """Ensure the charts directory exists."""
    if not os.path.exists('charts'):
        os.makedirs('charts')


def create_histogram_pandas(df, feature, bins=30, output_path=None):
    """
    Create histogram using pandas hist() method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    feature : str
        Name of the feature to plot
    bins : int, optional
        Number of bins for the histogram (default: 30)
    output_path : str, optional
        Path to save the chart (default: charts/{feature}_hist_pandas_bins{bins}.png)
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/{feature}_hist_pandas_bins{bins}.png'
    
    plt.figure(figsize=(10, 6))
    df[feature].hist(bins=bins, edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of {feature} (pandas hist(), bins={bins})', fontsize=14, fontweight='bold')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Use 150 DPI for PDF - sufficient quality and smaller file size
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_histogram_seaborn(df, feature, bins='auto', kde=False, output_path=None):
    """
    Create histogram using seaborn histplot() method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    feature : str
        Name of the feature to plot
    bins : int, str, or array, optional
        Number of bins or binning strategy (default: 'auto')
    kde : bool, optional
        Whether to plot a kernel density estimate (default: False)
    output_path : str, optional
        Path to save the chart (default: charts/{feature}_histplot_seaborn_bins{bins}.png)
    """
    ensure_charts_directory()
    
    if output_path is None:
        bins_str = str(bins).replace(' ', '_').replace('/', '_')
        kde_suffix = '_kde' if kde else ''
        output_path = f'charts/{feature}_histplot_seaborn_bins{bins_str}{kde_suffix}.png'
    
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=feature, bins=bins, kde=kde, edgecolor='black', alpha=0.7)
    bins_label = f'bins={bins}' if isinstance(bins, (int, str)) else f'bins={len(bins)-1}'
    kde_label = ' with KDE' if kde else ''
    plt.title(f'Histogram of {feature} (seaborn histplot(), {bins_label}{kde_label})', 
              fontsize=14, fontweight='bold')
    plt.xlabel(feature, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # Use 150 DPI for PDF - sufficient quality and smaller file size
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_histogram_comparison(df, feature, bin_values=[10, 30]):
    """
    Create multiple histograms with different bin values for comparison.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    feature : str
        Name of the feature to plot
    bin_values : list, optional
        List of bin values to try (default: [10, 30, 50, 100])
    
    Returns:
    --------
    list : List of output file paths
    """
    ensure_charts_directory()
    output_paths = []
    
    # Create pandas histograms with different bins
    for bins in bin_values:
        path = create_histogram_pandas(df, feature, bins=bins)
        output_paths.append(path)
    
    # Create seaborn histplots with different bins
    for bins in bin_values:
        path = create_histogram_seaborn(df, feature, bins=bins)
        output_paths.append(path)
    
    # Create seaborn histplot with auto bins and KDE
    path = create_histogram_seaborn(df, feature, bins='auto', kde=True)
    output_paths.append(path)
    
    return output_paths


def analyze_distribution(df, feature):
    """
    Analyze the distribution of a feature and return conclusions.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    feature : str
        Name of the feature to analyze
    
    Returns:
    --------
    dict : Dictionary containing distribution analysis
    """
    data = df[feature].dropna()
    
    # Calculate statistics
    mean = data.mean()
    median = data.median()
    std = data.std()
    skewness = data.skew()
    kurtosis = data.kurtosis()
    
    # Determine distribution type
    if abs(skewness) < 0.5:
        distribution_type = "approximately normal"
    elif skewness > 0.5:
        distribution_type = "right-skewed (positively skewed)"
    else:
        distribution_type = "left-skewed (negatively skewed)"
    
    # Additional analysis
    if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
        conclusion = "The distribution appears to be approximately normal."
    elif skewness > 1:
        conclusion = "The distribution is highly right-skewed, indicating a long tail on the right."
    elif skewness < -1:
        conclusion = "The distribution is highly left-skewed, indicating a long tail on the left."
    else:
        conclusion = f"The distribution is {distribution_type}."
    
    return {
        'feature': feature,
        'mean': mean,
        'median': median,
        'std': std,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'distribution_type': distribution_type,
        'conclusion': conclusion
    }


def generate_all_histograms(df, features=['age_years', 'weight', 'height']):
    """
    Generate all histograms for specified features using both methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    features : list, optional
        List of features to plot (default: ['age_years', 'weight', 'height'])
    
    Returns:
    --------
    dict : Dictionary mapping features to their chart paths and analysis
    """
    ensure_charts_directory()
    results = {}
    
    bin_values = [10, 30]
    
    for feature in features:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in dataframe. Skipping.")
            continue
        
        print(f"Generating histograms for {feature}...")
        
        # Generate comparison histograms
        chart_paths = create_histogram_comparison(df, feature, bin_values)
        
        # Analyze distribution
        analysis = analyze_distribution(df, feature)
        
        results[feature] = {
            'charts': chart_paths,
            'analysis': analysis
        }
    
    return results


def add_histogram_section_to_pdf(story, df, features=['age_years', 'weight', 'height']):
    """
    Add histogram section to PDF report with charts and analysis.
    
    Parameters:
    -----------
    story : list
        The PDF story list
    df : pandas.DataFrame
        The dataframe containing the data
    features : list, optional
        List of features to plot (default: ['age_years', 'weight', 'height'])
    """
    styles = getSampleStyleSheet()
    
    # Add heading
    heading = Paragraph("Histogram Analysis", styles['Heading1'])
    story.append(heading)
    story.append(Spacer(1, 0.2*inch))
    
    # Generate all histograms
    results = generate_all_histograms(df, features)
    
    for feature in features:
        if feature not in results:
            continue
        
        # Add feature heading
        feature_heading = Paragraph(f"Analysis of {feature}", styles['Heading2'])
        story.append(feature_heading)
        story.append(Spacer(1, 0.1*inch))
        
        # Add distribution analysis
        analysis = results[feature]['analysis']
        analysis_text = f"""
        <b>Distribution Analysis:</b><br/>
        Mean: {analysis['mean']:.2f}<br/>
        Median: {analysis['median']:.2f}<br/>
        Standard Deviation: {analysis['std']:.2f}<br/>
        Skewness: {analysis['skewness']:.2f}<br/>
        Kurtosis: {analysis['kurtosis']:.2f}<br/>
        Distribution Type: {analysis['distribution_type']}<br/><br/>
        <b>Conclusion:</b> {analysis['conclusion']}
        """
        para = Paragraph(analysis_text, styles['Normal'])
        story.append(para)
        story.append(Spacer(1, 0.15*inch))
        
        # Add comparison charts
        charts = results[feature]['charts']
        
        # Add pandas histograms
        subheading = Paragraph("Pandas hist() Method - Different Bin Values", styles['Heading3'])
        story.append(subheading)
        story.append(Spacer(1, 0.1*inch))
        
        # Add first few pandas charts (bins comparison)
        pandas_charts = [c for c in charts if 'pandas' in c]
        for chart_path in pandas_charts[:2]:  # Show 2 examples
            add_image_to_story(story, chart_path, width=5*inch)
        
        # Add seaborn histplots
        subheading = Paragraph("Seaborn histplot() Method - Different Bin Values", styles['Heading3'])
        story.append(subheading)
        story.append(Spacer(1, 0.1*inch))
        
        # Add first few seaborn charts (bins comparison)
        seaborn_charts = [c for c in charts if 'seaborn' in c and 'auto' not in c]
        for chart_path in seaborn_charts[:2]:  # Show 2 examples
            add_image_to_story(story, chart_path, width=5*inch)
        
        # Add seaborn with KDE
        subheading = Paragraph("Seaborn histplot() with KDE", styles['Heading3'])
        story.append(subheading)
        story.append(Spacer(1, 0.1*inch))
        
        kde_charts = [c for c in charts if 'kde' in c.lower() or 'auto' in c]
        if kde_charts:
            add_image_to_story(story, kde_charts[0], width=5*inch)
        
        story.append(Spacer(1, 0.3*inch))
    
    # Add overall conclusion
    conclusion_heading = Paragraph("Overall Distribution Conclusions", styles['Heading2'])
    story.append(conclusion_heading)
    story.append(Spacer(1, 0.1*inch))
    

