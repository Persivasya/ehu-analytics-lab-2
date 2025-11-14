import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.platypus import Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import numpy as np

from pdf_functions import add_image_to_story, add_heading, add_text


def ensure_charts_directory():
    """Ensure the charts directory exists."""
    if not os.path.exists('charts'):
        os.makedirs('charts')


def create_scatterplot_pandas(df, x_feature, y_feature, output_path=None):
    """
    Create scatter plot using pandas plot.scatter() method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x_feature : str
        Name of the feature for x-axis
    y_feature : str
        Name of the feature for y-axis
    output_path : str, optional
        Path to save the chart (default: charts/{x_feature}_vs_{y_feature}_scatter_pandas.png)
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/{x_feature}_vs_{y_feature}_scatter_pandas.png'
    
    plt.figure(figsize=(10, 6))
    df.plot.scatter(x=x_feature, y=y_feature, alpha=0.6, s=20)
    plt.title(f'Scatter Plot: {y_feature} vs {x_feature} (pandas scatter)', 
              fontsize=14, fontweight='bold')
    plt.xlabel(x_feature, fontsize=12)
    plt.ylabel(y_feature, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_scatterplot_seaborn(df, x_feature, y_feature, output_path=None):
    """
    Create scatter plot using seaborn scatterplot() method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x_feature : str
        Name of the feature for x-axis
    y_feature : str
        Name of the feature for y-axis
    output_path : str, optional
        Path to save the chart (default: charts/{x_feature}_vs_{y_feature}_scatter_seaborn.png)
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/{x_feature}_vs_{y_feature}_scatter_seaborn.png'
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_feature, y=y_feature, alpha=0.6, s=20)
    plt.title(f'Scatter Plot: {y_feature} vs {x_feature} (seaborn scatterplot)', 
              fontsize=14, fontweight='bold')
    plt.xlabel(x_feature, fontsize=12)
    plt.ylabel(y_feature, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_scatterplot_matplotlib(df, x_feature, y_feature, output_path=None):
    """
    Create scatter plot using matplotlib scatter() method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x_feature : str
        Name of the feature for x-axis
    y_feature : str
        Name of the feature for y-axis
    output_path : str, optional
        Path to save the chart (default: charts/{x_feature}_vs_{y_feature}_scatter_matplotlib.png)
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/{x_feature}_vs_{y_feature}_scatter_matplotlib.png'
    
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x_feature], df[y_feature], alpha=0.6, s=20)
    plt.title(f'Scatter Plot: {y_feature} vs {x_feature} (matplotlib scatter)', 
              fontsize=14, fontweight='bold')
    plt.xlabel(x_feature, fontsize=12)
    plt.ylabel(y_feature, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def calculate_correlation(df, x_feature, y_feature):
    """
    Calculate correlation coefficient between two features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x_feature : str
        Name of the first feature
    y_feature : str
        Name of the second feature
    
    Returns:
    --------
    float : Pearson correlation coefficient
    """
    data = df[[x_feature, y_feature]].dropna()
    correlation = data[x_feature].corr(data[y_feature])
    return correlation


def generate_scatterplots(df, plot_pairs=[('BMI', 'weight'), ('weight', 'height')]):
    """
    Generate scatter plots for specified feature pairs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    plot_pairs : list of tuples, optional
        List of (x_feature, y_feature) pairs to plot (default: [('BMI', 'weight'), ('weight', 'height')])
    
    Returns:
    --------
    dict : Dictionary mapping plot pairs to their chart paths and correlation
    """
    ensure_charts_directory()
    results = {}
    
    for x_feature, y_feature in plot_pairs:
        if x_feature not in df.columns or y_feature not in df.columns:
            print(f"Warning: Features '{x_feature}' or '{y_feature}' not found in dataframe. Skipping.")
            continue
        
        print(f"Generating scatter plots for {y_feature} vs {x_feature}...")
        
        # Calculate correlation
        correlation = calculate_correlation(df, x_feature, y_feature)
        
        chart_paths = {
            'seaborn': create_scatterplot_seaborn(df, x_feature, y_feature),
        }
        
        results[(x_feature, y_feature)] = {
            'charts': chart_paths,
            'correlation': correlation
        }
    
    return results


def add_scatterplot_section_to_pdf(story, df, plot_pairs=[('BMI', 'weight'), ('weight', 'height')]):
    """
    Add scatter plot section to PDF report with charts and correlation analysis.
    
    Parameters:
    -----------
    story : list
        The PDF story list
    df : pandas.DataFrame
        The dataframe containing the data
    plot_pairs : list of tuples, optional
        List of (x_feature, y_feature) pairs to plot (default: [('BMI', 'weight'), ('weight', 'height')])
    """
    styles = getSampleStyleSheet()
    
    # Add heading
    heading = Paragraph("Scatter Plot Analysis", styles['Heading1'])
    story.append(heading)
    story.append(Spacer(1, 0.2*inch))
    
    # Generate all scatter plots
    results = generate_scatterplots(df, plot_pairs)
    
    for x_feature, y_feature in plot_pairs:
        plot_key = (x_feature, y_feature)
        if plot_key not in results:
            continue
        
        # Add plot pair heading
        plot_heading = Paragraph(f"Scatter Plot: {y_feature} vs {x_feature}", styles['Heading2'])
        story.append(plot_heading)
        story.append(Spacer(1, 0.1*inch))
        
        # Calculate and display correlation
        correlation = results[plot_key]['correlation']
        
        correlation_text = f"""
        <b>Correlation Analysis:</b><br/>
        Pearson correlation coefficient: {correlation:.4f}<br/>
        """
        
        # Add interpretation
        if abs(correlation) >= 0.7:
            strength = "strong"
        elif abs(correlation) >= 0.4:
            strength = "moderate"
        else:
            strength = "weak"
        
        if correlation > 0:
            direction = "positive"
        else:
            direction = "negative"
        
        correlation_text += f"""
        Interpretation: {strength.capitalize()} {direction} correlation<br/>
        """
        
        para = Paragraph(correlation_text, styles['Normal'])
        story.append(para)
        story.append(Spacer(1, 0.15*inch))
        
        # Add scatter plots
        charts = results[plot_key]['charts']
        
        # Add seaborn scatterplot
        subheading = Paragraph("Seaborn scatterplot() Method", styles['Heading3'])
        story.append(subheading)
        story.append(Spacer(1, 0.1*inch))
        add_image_to_story(story, charts['seaborn'], width=5*inch)
        
        story.append(Spacer(1, 0.3*inch))
    

