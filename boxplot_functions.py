import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from pdf_functions import add_image_to_story, add_heading, add_text


def ensure_charts_directory():
    """Ensure the charts directory exists."""
    if not os.path.exists('charts'):
        os.makedirs('charts')


def calculate_quantiles(df, feature):
    """
    Calculate Q1, Q2 (median), and Q3 quantiles for a feature.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    feature : str
        Name of the feature to calculate quantiles for
    
    Returns:
    --------
    dict : Dictionary containing Q1, Q2 (median), and Q3 values
    """
    data = df[feature].dropna()
    
    q1 = data.quantile(0.25)
    q2 = data.quantile(0.50)  # Median
    q3 = data.quantile(0.75)
    iqr = q3 - q1
    
    # Calculate outliers (values beyond Q1 - 1.5*IQR and Q3 + 1.5*IQR)
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    num_outliers = len(outliers)
    
    return {
        'Q1': q1,
        'Q2': q2,  # Median
        'Q3': q3,
        'IQR': iqr,  # Interquartile Range
        'min': data.min(),
        'max': data.max(),
        'num_outliers': num_outliers,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }


def create_boxplot_seaborn(df, feature, num_outliers=0, output_path=None):
    """
    Create box plot using seaborn boxplot() method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    feature : str
        Name of the feature to plot
    num_outliers : int, optional
        Number of outliers to display in the title (default: 0)
    output_path : str, optional
        Path to save the chart (default: charts/{feature}_boxplot_seaborn.png)
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/{feature}_boxplot_seaborn.png'
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, y=feature)
    
    # Add title with outlier count
    title = f'Box Plot of {feature} (seaborn boxplot())'
    if num_outliers > 0:
        title += f'\nNumber of outliers: {num_outliers}'
    else:
        title += '\nNo outliers detected'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel(feature, fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path



def generate_boxplots(df, features=['weight', 'height']):
    """
    Generate box plots for specified features using different methods.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    features : list, optional
        List of features to plot (default: ['weight', 'height'])
    
    Returns:
    --------
    dict : Dictionary mapping features to their chart paths and outlier counts
    """
    ensure_charts_directory()
    results = {}
    
    for feature in features:
        if feature not in df.columns:
            print(f"Warning: Feature '{feature}' not found in dataframe. Skipping.")
            continue
        
        print(f"Generating box plots for {feature}...")
        
        # Calculate quantiles to get outlier count
        quantiles = calculate_quantiles(df, feature)
        num_outliers = quantiles['num_outliers']
        
        chart_paths = {
            'seaborn': create_boxplot_seaborn(df, feature, num_outliers=num_outliers),
        }
        
        results[feature] = {
            'charts': chart_paths,
            'num_outliers': num_outliers
        }
    
    return results


def add_boxplot_section_to_pdf(story, df, features=['weight', 'height']):
    """
    Add box plot section to PDF report with charts and quantile analysis.
    
    Parameters:
    -----------
    story : list
        The PDF story list
    df : pandas.DataFrame
        The dataframe containing the data
    features : list, optional
        List of features to plot (default: ['weight', 'height'])
    """
    styles = getSampleStyleSheet()
    
    # Add heading
    heading = Paragraph("Box Plot Analysis", styles['Heading1'])
    story.append(heading)
    story.append(Spacer(1, 0.2*inch))
    
    # Generate all box plots
    results = generate_boxplots(df, features)
    
    for feature in features:
        if feature not in results:
            continue
        
        # Add feature heading
        feature_heading = Paragraph(f"Analysis of {feature}", styles['Heading2'])
        story.append(feature_heading)
        story.append(Spacer(1, 0.1*inch))
        
        # Calculate and display quantiles
        quantiles = calculate_quantiles(df, feature)
        num_outliers = results[feature].get('num_outliers', quantiles['num_outliers'])
        
        quantiles_text = f"""
        <b>Quantile Analysis:</b><br/>
        Q1 (25th percentile): {quantiles['Q1']:.2f}<br/>
        Q2 (Median, 50th percentile): {quantiles['Q2']:.2f}<br/>
        Q3 (75th percentile): {quantiles['Q3']:.2f}<br/>
        Interquartile Range (IQR): {quantiles['IQR']:.2f}<br/>
        Minimum: {quantiles['min']:.2f}<br/>
        Maximum: {quantiles['max']:.2f}<br/>
        <b>Number of outliers:</b> {num_outliers} (values beyond Q1 - 1.5×IQR or Q3 + 1.5×IQR)<br/>
        """
        para = Paragraph(quantiles_text, styles['Normal'])
        story.append(para)
        story.append(Spacer(1, 0.15*inch))
        
        # Add box plots
        charts = results[feature]['charts']
        
        # Add seaborn boxplot
        subheading = Paragraph("Seaborn boxplot() Method", styles['Heading3'])
        story.append(subheading)
        story.append(Spacer(1, 0.1*inch))
        add_image_to_story(story, charts['seaborn'], width=5*inch)

        story.append(Spacer(1, 0.3*inch))
    
