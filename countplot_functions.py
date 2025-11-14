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


def create_countplot_seaborn(df, age_column, cardio_column, output_path=None):
    """
    Create countplot with age on X-axis and number of people on Y-axis,
    with two columns for each age corresponding to each cardio class.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str
        Name of the age column (e.g., 'age_years')
    cardio_column : str
        Name of the cardio column (e.g., 'cardio')
    output_path : str, optional
        Path to save the chart (default: charts/countplot_age_cardio.png)
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/countplot_{age_column}_cardio.png'
    
    # Prepare data
    plot_data = df[[age_column, cardio_column]].copy().dropna()
    
    # Convert cardio to categorical for better labeling
    plot_data[cardio_column] = plot_data[cardio_column].astype('category')
    plot_data[cardio_column] = plot_data[cardio_column].cat.rename_categories({0: 'No Cardio Disease', 1: 'Cardio Disease'})
    
    plt.figure(figsize=(14, 6))
    
    # Create countplot with hue to split by cardio class
    sns.countplot(data=plot_data, x=age_column, hue=cardio_column, palette=['#3498db', '#e74c3c'])
    
    plt.title(f'Count Plot: Number of People by Age and Cardiovascular Disease Status', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Number of People', fontsize=12)
    plt.legend(title='Cardiovascular Disease', loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def calculate_cardio_percentages_by_age(df, age_column, cardio_column):
    """
    Calculate the percentage of individuals with and without cardiovascular disease for each age.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str
        Name of the age column
    cardio_column : str
        Name of the cardio column
    
    Returns:
    --------
    pandas.DataFrame : DataFrame with age, counts, and percentages for each cardio class
    """
    # Prepare data
    plot_data = df[[age_column, cardio_column]].copy().dropna()
    
    # Calculate counts by age and cardio
    counts = plot_data.groupby([age_column, cardio_column]).size().reset_index(name='count')
    
    # Calculate total counts per age
    total_by_age = plot_data.groupby(age_column).size().reset_index(name='total')
    
    # Merge to get percentages
    result = counts.merge(total_by_age, on=age_column)
    result['percentage'] = (result['count'] / result['total'] * 100).round(2)
    
    return result


def find_cardio_threshold_age(df, age_column, cardio_column):
    """
    Find the age at which the percentage of individuals with cardiovascular disease 
    surpasses those without it.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str
        Name of the age column
    cardio_column : str
        Name of the cardio column
    
    Returns:
    --------
    dict : Dictionary containing threshold age and analysis
    """
    # Calculate percentages
    percentages_df = calculate_cardio_percentages_by_age(df, age_column, cardio_column)
    
    # Separate data for cardio=0 and cardio=1
    cardio_0 = percentages_df[percentages_df[cardio_column] == 0].set_index(age_column)
    cardio_1 = percentages_df[percentages_df[cardio_column] == 1].set_index(age_column)
    
    # Find ages where cardio=1 percentage > cardio=0 percentage
    merged = cardio_0.merge(cardio_1, left_index=True, right_index=True, suffixes=('_no', '_yes'))
    threshold_ages = merged[merged['percentage_yes'] > merged['percentage_no']].index
    
    if len(threshold_ages) > 0:
        threshold_age = int(threshold_ages.min())
        
        # Get the exact percentages at threshold age
        threshold_data = merged.loc[threshold_age]
        
        result = {
            'threshold_age': threshold_age,
            'cardio_disease_percentage': float(threshold_data['percentage_yes']),
            'no_cardio_disease_percentage': float(threshold_data['percentage_no']),
            'found': True
        }
    else:
        # If no threshold found, find the age with closest percentages
        merged['diff'] = abs(merged['percentage_yes'] - merged['percentage_no'])
        closest_age = merged['diff'].idxmin()
        closest_data = merged.loc[closest_age]
        
        result = {
            'threshold_age': int(closest_age),
            'cardio_disease_percentage': float(closest_data['percentage_yes']),
            'no_cardio_disease_percentage': float(closest_data['percentage_no']),
            'found': False,
            'note': 'No age found where cardio disease percentage exceeds no cardio disease. Showing closest match.'
        }
    
    return result


def create_percentage_plot(df, age_column, cardio_column, output_path=None):
    """
    Create a line plot showing the percentage of individuals with and without 
    cardiovascular disease by age.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str
        Name of the age column
    cardio_column : str
        Name of the cardio column
    output_path : str, optional
        Path to save the chart
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/percentage_plot_{age_column}_cardio.png'
    
    # Calculate percentages
    percentages_df = calculate_cardio_percentages_by_age(df, age_column, cardio_column)
    
    # Separate data for cardio=0 and cardio=1
    cardio_0 = percentages_df[percentages_df[cardio_column] == 0].sort_values(age_column)
    cardio_1 = percentages_df[percentages_df[cardio_column] == 1].sort_values(age_column)
    
    plt.figure(figsize=(12, 6))
    
    # Plot lines for both groups
    plt.plot(cardio_0[age_column], cardio_0['percentage'], 
             marker='o', label='No Cardio Disease', linewidth=2, markersize=4)
    plt.plot(cardio_1[age_column], cardio_1['percentage'], 
             marker='s', label='Cardio Disease', linewidth=2, markersize=4)
    
    # Find and mark threshold age
    threshold_info = find_cardio_threshold_age(df, age_column, cardio_column)
    if threshold_info['found']:
        threshold_age = threshold_info['threshold_age']
        plt.axvline(x=threshold_age, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold Age: {threshold_age}')
        plt.text(threshold_age, plt.ylim()[1] * 0.9, f'Age {threshold_age}', 
                rotation=90, verticalalignment='top', fontsize=10, fontweight='bold')
    
    plt.title('Percentage of Individuals with/without Cardiovascular Disease by Age', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_countplot_analysis(df, age_column='age_years', cardio_column='cardio'):
    """
    Generate countplot and analysis for age vs cardiovascular disease.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str, optional
        Name of the age column (default: 'age_years')
    cardio_column : str, optional
        Name of the cardio column (default: 'cardio')
    
    Returns:
    --------
    dict : Dictionary containing chart paths and analysis results
    """
    ensure_charts_directory()
    
    if age_column not in df.columns or cardio_column not in df.columns:
        print(f"Warning: Columns '{age_column}' or '{cardio_column}' not found in dataframe.")
        return None
    
    print(f"Generating countplot analysis for {age_column} vs {cardio_column}...")
    
    # Create countplot
    countplot_path = create_countplot_seaborn(df, age_column, cardio_column)
    
    # Create percentage plot
    percentage_plot_path = create_percentage_plot(df, age_column, cardio_column)
    
    # Find threshold age
    threshold_info = find_cardio_threshold_age(df, age_column, cardio_column)
    
    # Calculate overall statistics
    percentages_df = calculate_cardio_percentages_by_age(df, age_column, cardio_column)
    
    return {
        'countplot_path': countplot_path,
        'percentage_plot_path': percentage_plot_path,
        'threshold_info': threshold_info,
        'percentages_df': percentages_df
    }


def add_countplot_section_to_pdf(story, df, age_column='age_years', cardio_column='cardio'):
    """
    Add countplot section to PDF report with charts and threshold analysis.
    
    Parameters:
    -----------
    story : list
        The PDF story list
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str, optional
        Name of the age column (default: 'age_years')
    cardio_column : str, optional
        Name of the cardio column (default: 'cardio')
    """
    styles = getSampleStyleSheet()
    
    # Add heading
    heading = Paragraph("Count Plot Analysis: Age vs Cardiovascular Disease", styles['Heading1'])
    story.append(heading)
    story.append(Spacer(1, 0.2*inch))
    
    # Generate analysis
    results = generate_countplot_analysis(df, age_column, cardio_column)
    
    if results is None:
        error_text = f"Error: Could not generate countplot analysis. Please check column names."
        para = Paragraph(error_text, styles['Normal'])
        story.append(para)
        return
    
    # Add countplot
    subheading = Paragraph("Count Plot: Number of People by Age and Cardio Disease Status", styles['Heading2'])
    story.append(subheading)
    story.append(Spacer(1, 0.1*inch))
    
    description_text = """
    The count plot below shows the number of people for each age, with two columns per age 
    representing the count of people with and without cardiovascular disease.
    """
    para = Paragraph(description_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    add_image_to_story(story, results['countplot_path'], width=5*inch)
    
    # Add threshold analysis
    threshold_heading = Paragraph("Threshold Age Analysis", styles['Heading2'])
    story.append(threshold_heading)
    story.append(Spacer(1, 0.1*inch))
    
    threshold_info = results['threshold_info']
    
    
    # Add percentage plot
    percentage_heading = Paragraph("Percentage Plot: Cardiovascular Disease by Age", styles['Heading2'])
    story.append(percentage_heading)
    story.append(Spacer(1, 0.1*inch))
    
    
    add_image_to_story(story, results['percentage_plot_path'], width=5*inch)
    
