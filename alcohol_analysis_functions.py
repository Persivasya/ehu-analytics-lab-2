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


def create_generalized_countplot_seaborn(df, x_column, hue_column, output_path=None, 
                                        title=None, x_label=None, hue_label=None, 
                                        palette=None, figsize=(14, 6)):
    """
    Generalized countplot function based on create_countplot_seaborn from countplot_functions.py.
    Creates a countplot with x_column on X-axis and number of people on Y-axis,
    with columns for each value of hue_column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    x_column : str
        Name of the column for X-axis (e.g., 'age_years')
    hue_column : str
        Name of the column to split by (e.g., 'gender', 'alco')
    output_path : str, optional
        Path to save the chart
    title : str, optional
        Title for the plot (default: auto-generated)
    x_label : str, optional
        Label for X-axis (default: x_column)
    hue_label : str, optional
        Label for hue legend (default: hue_column)
    palette : list, optional
        Color palette for the plot (default: seaborn default)
    figsize : tuple, optional
        Figure size (default: (14, 6))
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/countplot_{x_column}_by_{hue_column}.png'
    
    # Prepare data
    plot_data = df[[x_column, hue_column]].copy().dropna()
    
    # Convert hue column to categorical for better labeling
    plot_data[hue_column] = plot_data[hue_column].astype('category')
    
    plt.figure(figsize=figsize)
    
    # Create countplot with hue
    if palette:
        sns.countplot(data=plot_data, x=x_column, hue=hue_column, palette=palette)
    else:
        sns.countplot(data=plot_data, x=x_column, hue=hue_column)
    
    # Set title
    if title is None:
        title = f'Count Plot: Number of People by {x_column} and {hue_column}'
    plt.title(title, fontsize=14, fontweight='bold')
    
    # Set labels
    if x_label is None:
        x_label = x_column.replace('_', ' ').title()
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Number of People', fontsize=12)
    
    # Set legend
    if hue_label is None:
        hue_label = hue_column.replace('_', ' ').title()
    plt.legend(title=hue_label, loc='upper right')
    
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_alcohol_by_gender_countplot(df, age_column='age_years', gender_column='gender', 
                                     alcohol_column='alco', output_path=None):
    """
    Create countplot showing alcohol consumption by gender and age.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str, optional
        Name of the age column (default: 'age_years')
    gender_column : str, optional
        Name of the gender column (default: 'gender')
    alcohol_column : str, optional
        Name of the alcohol column (default: 'alco')
    output_path : str, optional
        Path to save the chart
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = f'charts/alcohol_consumption_by_gender_age.png'
    
    # Prepare data
    plot_data = df[[age_column, gender_column, alcohol_column]].copy().dropna()
    
    # Filter only those who consume alcohol (alco == 1)
    alcohol_consumers = plot_data[plot_data[alcohol_column] == 1].copy()
    
    # Map gender values to labels (assuming 1=male, 2=female, adjust if needed)
    gender_mapping = {1: 'Male', 2: 'Female'}
    alcohol_consumers[gender_column] = alcohol_consumers[gender_column].map(gender_mapping)
    
    plt.figure(figsize=(14, 6))
    
    # Create countplot with age on X-axis and gender as hue
    sns.countplot(data=alcohol_consumers, x=age_column, hue=gender_column, 
                 palette=['#3498db', '#e74c3c'])
    
    plt.title('Alcohol Consumption by Gender and Age (Count Plot)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Number of People Consuming Alcohol', fontsize=12)
    plt.legend(title='Gender', loc='upper right')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def calculate_alcohol_consumption_stats(df, gender_column='gender', alcohol_column='alco'):
    """
    Calculate statistics about alcohol consumption by gender.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    gender_column : str, optional
        Name of the gender column (default: 'gender')
    alcohol_column : str, optional
        Name of the alcohol column (default: 'alco')
    
    Returns:
    --------
    dict : Dictionary containing statistics for each gender
    """
    # Prepare data
    plot_data = df[[gender_column, alcohol_column]].copy().dropna()
    
    # Map gender values to labels
    gender_mapping = {1: 'Male', 2: 'Female'}
    plot_data[gender_column] = plot_data[gender_column].map(gender_mapping)
    
    stats = {}
    for gender in ['Male', 'Female']:
        gender_data = plot_data[plot_data[gender_column] == gender]
        total = len(gender_data)
        alcohol_consumers = len(gender_data[gender_data[alcohol_column] == 1])
        non_consumers = total - alcohol_consumers
        percentage = (alcohol_consumers / total * 100) if total > 0 else 0
        
        stats[gender] = {
            'total': total,
            'alcohol_consumers': alcohol_consumers,
            'non_consumers': non_consumers,
            'percentage': percentage
        }
    
    # Determine who consumes more
    if stats['Male']['percentage'] > stats['Female']['percentage']:
        answer = 'Men'
        difference = stats['Male']['percentage'] - stats['Female']['percentage']
    elif stats['Female']['percentage'] > stats['Male']['percentage']:
        answer = 'Women'
        difference = stats['Female']['percentage'] - stats['Male']['percentage']
    else:
        answer = 'Equal'
        difference = 0
    
    stats['answer'] = answer
    stats['difference'] = difference
    
    return stats


def generate_alcohol_analysis(df, age_column='age_years', gender_column='gender', 
                            alcohol_column='alco'):
    """
    Generate alcohol consumption analysis including countplot and statistics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str, optional
        Name of the age column (default: 'age_years')
    gender_column : str, optional
        Name of the gender column (default: 'gender')
    alcohol_column : str, optional
        Name of the alcohol column (default: 'alco')
    
    Returns:
    --------
    dict : Dictionary containing chart paths and statistics
    """
    ensure_charts_directory()
    
    if age_column not in df.columns or gender_column not in df.columns or alcohol_column not in df.columns:
        print(f"Warning: Required columns not found in dataframe.")
        return None
    
    print(f"Generating alcohol consumption analysis...")
    
    # Create countplot
    countplot_path = create_alcohol_by_gender_countplot(df, age_column, gender_column, alcohol_column)
    
    # Calculate statistics
    stats = calculate_alcohol_consumption_stats(df, gender_column, alcohol_column)
    
    return {
        'countplot_path': countplot_path,
        'statistics': stats
    }


def add_alcohol_analysis_section_to_pdf(story, df, age_column='age_years', 
                                       gender_column='gender', alcohol_column='alco'):
    """
    Add alcohol consumption analysis section to PDF report with charts and statistics.
    
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
    alcohol_column : str, optional
        Name of the alcohol column (default: 'alco')
    """
    styles = getSampleStyleSheet()
    
    # Add heading
    heading = Paragraph("Alcohol Consumption Analysis: Gender Comparison", styles['Heading1'])
    story.append(heading)
    story.append(Spacer(1, 0.2*inch))
    
    # Generate analysis
    results = generate_alcohol_analysis(df, age_column, gender_column, alcohol_column)
    
    if results is None:
        error_text = "Error: Could not generate alcohol consumption analysis. Please check column names."
        para = Paragraph(error_text, styles['Normal'])
        story.append(para)
        return
    
    # Add question and answer
    question_heading = Paragraph("Research Question", styles['Heading2'])
    story.append(question_heading)
    story.append(Spacer(1, 0.1*inch))
    
    question_text = "Who more often report consuming alcohol – men or women?"
    para = Paragraph(f"<b>Question:</b> {question_text}", styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    # Display statistics and answer
    stats = results['statistics']
    
    answer_text = f"""
    <b>Answer:</b> {stats['answer']} more often report consuming alcohol.<br/><br/>
    
    <b>Statistics by Gender:</b><br/>
    """
    
    for gender in ['Male', 'Female']:
        gender_stats = stats[gender]
        answer_text += f"""
        <b>{gender}:</b><br/>
        Total individuals: {gender_stats['total']:,}<br/>
        Alcohol consumers: {gender_stats['alcohol_consumers']:,} ({gender_stats['percentage']:.2f}%)<br/>
        Non-consumers: {gender_stats['non_consumers']:,} ({100 - gender_stats['percentage']:.2f}%)<br/><br/>
        """
    
    if stats['difference'] > 0:
        answer_text += f"""
        <b>Difference:</b> {stats['answer']} have {stats['difference']:.2f} percentage points higher 
        alcohol consumption rate than the other gender.<br/>
        """
    
    para = Paragraph(answer_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.15*inch))
    
    # Add countplot
    plot_heading = Paragraph("Count Plot: Alcohol Consumption by Gender and Age", styles['Heading2'])
    story.append(plot_heading)
    story.append(Spacer(1, 0.1*inch))
    
    description_text = """
    The count plot below shows the number of people who consume alcohol, divided by gender, 
    with age on the X-axis. This visualization helps identify age-related patterns in 
    alcohol consumption between men and women.
    """
    para = Paragraph(description_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    add_image_to_story(story, results['countplot_path'], width=5*inch)
    
    # Add conclusion
    conclusion_heading = Paragraph("Alcohol Consumption Analysis Conclusions", styles['Heading2'])
    story.append(conclusion_heading)
    story.append(Spacer(1, 0.1*inch))
    
    conclusion_text = """
    <b>Key Findings:</b><br/>
    • The count plot reveals the distribution of alcohol consumption across different age groups for each gender.<br/>
    • By comparing the heights of the bars for men and women at each age, we can see which gender has higher 
    alcohol consumption rates at different life stages.<br/>
    • The overall statistics show the total percentage of each gender that reports consuming alcohol.<br/>
    • This analysis helps understand gender-specific patterns in alcohol consumption behavior.<br/><br/>
    
    <b>Methodology:</b><br/>
    • The analysis uses a generalized countplot function based on create_countplot_seaborn from countplot_functions.py.<br/>
    • The plot uses age as the X-axis and divides the data by gender using the hue parameter.<br/>
    • Only individuals who report consuming alcohol (alco=1) are included in the visualization.
    """
    para = Paragraph(conclusion_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.2*inch))

