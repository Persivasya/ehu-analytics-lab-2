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


def prepare_data_for_violinplot(df, value_column, group_column):
    """
    Prepare data for violin plot using melt() to convert to long format.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    value_column : str
        Name of the column with values to plot (e.g., 'height')
    group_column : str
        Name of the column to group by (e.g., 'gender')
    
    Returns:
    --------
    pandas.DataFrame : Dataframe in long format suitable for violin plots
    """
    # Select relevant columns
    plot_data = df[[value_column, group_column]].copy()
    
    # Drop rows with missing values
    plot_data = plot_data.dropna()
    
    # To demonstrate melt(), we'll create a wide format first, then use melt() to convert to long format
    # Create a wide format by pivoting: each gender becomes a column
    wide_data = plot_data.pivot_table(
        values=value_column,
        index=plot_data.groupby([group_column]).cumcount(),
        columns=group_column,
        aggfunc='first'
    )
    
    # Reset index
    wide_data = wide_data.reset_index(drop=True)
    
    # Now use melt() to convert from wide format back to long format
    # This demonstrates the melt() function: converting wide format to long format
    melted_data = pd.melt(
        wide_data,
        id_vars=[],  # No identifier variables needed
        value_vars=list(wide_data.columns),  # All columns are value columns
        var_name=group_column,  # Name for the grouping variable
        value_name=value_column  # Name for the value variable
    )
    
    # Drop rows with missing values after melting
    melted_data = melted_data.dropna()
    
    # Return the melted data in long format
    return melted_data[[value_column, group_column]]

def map_gender(gender_column):
    return {1: 'Female', 2: 'Male'}

def create_violinplot_seaborn(df, value_column, group_column, output_path=None):
    """
    Create violin plot using seaborn violinplot() method.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    value_column : str
        Name of the column with values to plot (e.g., 'height')
    group_column : str
        Name of the column to group by (e.g., 'gender')
    output_path : str, optional
        Path to save the chart (default: charts/{value_column}_by_{group_column}_violinplot.png)
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/{value_column}_by_{group_column}_violinplot.png'
    
    # Prepare data using melt if needed, or use directly
    plot_data = prepare_data_for_violinplot(df, value_column, group_column)
    
    plt.figure(figsize=(10, 6))
    
    # Create violin plot with hue to split by group_column
    # scale='count' scales the width of each violin by the number of observations
    sns.violinplot(data=plot_data, x=group_column, y=value_column, 
                   hue=group_column, scale='count', inner='box')
    
    plt.title(f'Violin Plot: {value_column} by {group_column} (seaborn violinplot)', 
              fontsize=14, fontweight='bold')
    plt.xlabel(map_gender(group_column), fontsize=12)
    plt.ylabel(value_column, fontsize=12)
    
    # Add legend
    plt.legend(title=map_gender(group_column), loc='upper right')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def get_gender_statistics(df, value_column, group_column):
    """
    Get statistics for each group (e.g., count, mean, median for each gender).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    value_column : str
        Name of the column with values
    group_column : str
        Name of the column to group by
    
    Returns:
    --------
    dict : Dictionary with statistics for each group
    """
    plot_data = prepare_data_for_violinplot(df, value_column, group_column)
    
    stats = {}
    for group in plot_data[group_column].unique():
        group_data = plot_data[plot_data[group_column] == group][value_column]
        stats[group] = {
            'count': len(group_data),
            'mean': group_data.mean(),
            'median': group_data.median(),
            'std': group_data.std(),
            'min': group_data.min(),
            'max': group_data.max()
        }
    
    return stats


def generate_violinplots(df, plot_configs=[('height', 'gender')]):
    """
    Generate violin plots for specified value and group column pairs.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    plot_configs : list of tuples, optional
        List of (value_column, group_column) pairs to plot (default: [('height', 'gender')])
    
    Returns:
    --------
    dict : Dictionary mapping plot configs to their chart paths and statistics
    """
    ensure_charts_directory()
    results = {}
    
    for value_column, group_column in plot_configs:
        if value_column not in df.columns or group_column not in df.columns:
            print(f"Warning: Columns '{value_column}' or '{group_column}' not found in dataframe. Skipping.")
            continue
        
        print(f"Generating violin plot for {value_column} by {group_column}...")
        
        # Get statistics
        stats = get_gender_statistics(df, value_column, group_column)
        
        # Create violin plot
        chart_path = create_violinplot_seaborn(df, value_column, group_column)
        
        results[(value_column, group_column)] = {
            'chart': chart_path,
            'statistics': stats
        }
    
    return results


def add_violinplot_section_to_pdf(story, df, plot_configs=[('height', 'gender')]):
    """
    Add violin plot section to PDF report with charts and statistics.
    
    Parameters:
    -----------
    story : list
        The PDF story list
    df : pandas.DataFrame
        The dataframe containing the data
    plot_configs : list of tuples, optional
        List of (value_column, group_column) pairs to plot (default: [('height', 'gender')])
    """
    styles = getSampleStyleSheet()
    
    # Add heading
    heading = Paragraph("Violin Plot Analysis", styles['Heading1'])
    story.append(heading)
    story.append(Spacer(1, 0.2*inch))
    
    # Generate all violin plots
    results = generate_violinplots(df, plot_configs)
    
    for value_column, group_column in plot_configs:
        plot_key = (value_column, group_column)
        if plot_key not in results:
            continue
        
        # Add plot heading
        plot_heading = Paragraph(f"Violin Plot: {value_column} by {group_column}", styles['Heading2'])
        story.append(plot_heading)
        story.append(Spacer(1, 0.1*inch))
        
        # Display statistics for each group
        stats = results[plot_key]['statistics']
        
        stats_text = f"<b>Statistics by {group_column}:</b><br/>"
        for group, group_stats in stats.items():
            stats_text += f"""
            <b>{group_column} = {group}:</b><br/>
            Count: {group_stats['count']}<br/>
            Mean: {group_stats['mean']:.2f}<br/>
            Median: {group_stats['median']:.2f}<br/>
            Standard Deviation: {group_stats['std']:.2f}<br/>
            Min: {group_stats['min']:.2f}<br/>
            Max: {group_stats['max']:.2f}<br/><br/>
            """
        
        para = Paragraph(stats_text, styles['Normal'])
        story.append(para)
        story.append(Spacer(1, 0.15*inch))
        
        # Add violin plot
        chart_path = results[plot_key]['chart']
        
        subheading = Paragraph("Seaborn violinplot() Method", styles['Heading3'])
        story.append(subheading)
        story.append(Spacer(1, 0.1*inch))
        
        # Add note about the plot
        note_text = f"""
        <i>Note: The violin plot shows the distribution of {value_column} for each {group_column} category. 
        The width of each violin is scaled by the number of records (scale='count'), and the plot uses 
        hue parameter to split by {group_column}.</i>
        """
        note_para = Paragraph(note_text, styles['Normal'])
        story.append(note_para)
        story.append(Spacer(1, 0.05*inch))
        
        add_image_to_story(story, chart_path, width=5*inch)
        
        story.append(Spacer(1, 0.3*inch))
    

