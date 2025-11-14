import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from reportlab.platypus import Spacer, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

from pdf_functions import add_image_to_story, add_heading, add_text


def ensure_charts_directory():
    """Ensure the charts directory exists."""
    if not os.path.exists('charts'):
        os.makedirs('charts')


def check_binary_balance(df, binary_column, group_column):
    """
    Check if a binary feature is balanced across groups.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    binary_column : str
        Name of the binary column to check
    group_column : str
        Name of the group column (e.g., 'gender')
    
    Returns:
    --------
    dict : Dictionary with balance statistics
    """
    plot_data = df[[binary_column, group_column]].copy().dropna()
    
    # Map gender values to labels
    if group_column == 'gender':
        plot_data[group_column] = plot_data[group_column].map({1: 'Female', 2: 'Male'})
    
    balance_stats = {}
    groups = plot_data[group_column].unique()
    
    for group in groups:
        group_data = plot_data[plot_data[group_column] == group]
        total = len(group_data)
        
        if total == 0:
            continue
        
        # Count values in binary column
        value_counts = group_data[binary_column].value_counts().to_dict()
        
        # Calculate proportions
        proportions = {k: v / total for k, v in value_counts.items()}
        
        balance_stats[group] = {
            'total': total,
            'value_counts': value_counts,
            'proportions': proportions
        }
    
    # Calculate balance metric (how close proportions are to 50/50)
    # For binary features, ideal balance is 50/50
    all_proportions = []
    for group_stats in balance_stats.values():
        all_proportions.extend(group_stats['proportions'].values())
    
    if all_proportions:
        max_prop = max(all_proportions)
        min_prop = min(all_proportions)
        balance_ratio = min_prop / max_prop if max_prop > 0 else 0
        
        # Determine if balanced (threshold: if ratio > 0.4, consider relatively balanced)
        is_balanced = balance_ratio > 0.4
    else:
        balance_ratio = 0
        is_balanced = False
    
    balance_stats['overall'] = {
        'balance_ratio': balance_ratio,
        'is_balanced': is_balanced
    }
    
    return balance_stats


def create_catplot_smoking_gender_age(df, age_column='age_years', gender_column='gender',
                                      smoke_column='smoke', output_path=None):
    """
    Create catplot showing smoking patterns by gender over age.
    
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
    output_path : str, optional
        Path to save the chart
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = 'charts/catplot_smoking_gender_age.png'
    
    # Prepare data
    plot_data = df[[age_column, gender_column, smoke_column]].copy().dropna()
    
    # Map gender values to labels
    plot_data[gender_column] = plot_data[gender_column].map({1: 'Female', 2: 'Male'})
    
    # Map smoking values to labels
    plot_data[smoke_column] = plot_data[smoke_column].map({0: 'Non-smoker', 1: 'Smoker'})
    
    # Create age bins for better visualization
    plot_data['age_bin'] = pd.cut(plot_data[age_column], bins=10, precision=0)
    plot_data['age_bin_mid'] = plot_data['age_bin'].apply(lambda x: int(x.mid))
    
    # Create catplot using seaborn
    # x: age bins, hue: gender, col: smoking status
    g = sns.catplot(
        data=plot_data,
        x='age_bin_mid',
        hue=gender_column,
        col=smoke_column,
        kind='count',
        height=6,
        aspect=1.2,
        palette=['#3498db', '#e74c3c']
    )
    
    g.fig.suptitle('Smoking Patterns by Gender Over Age (Catplot)', 
                   fontsize=14, fontweight='bold', y=1.02)
    g.set_axis_labels('Age (years)', 'Count')
    g.set_xticklabels(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_catplot_cholesterol_gender_age(df, age_column='age_years', gender_column='gender',
                                         cholesterol_column='cholesterol', output_path=None):
    """
    Create catplot showing cholesterol levels by gender over age.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    age_column : str, optional
        Name of the age column (default: 'age_years')
    gender_column : str, optional
        Name of the gender column (default: 'gender')
    cholesterol_column : str, optional
        Name of the cholesterol column (default: 'cholesterol')
    output_path : str, optional
        Path to save the chart
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = 'charts/catplot_cholesterol_gender_age.png'
    
    # Prepare data
    plot_data = df[[age_column, gender_column, cholesterol_column]].copy().dropna()
    
    # Map gender values to labels
    plot_data[gender_column] = plot_data[gender_column].map({1: 'Female', 2: 'Male'})
    
    # Map cholesterol values to labels
    cholesterol_mapping = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
    plot_data[cholesterol_column] = plot_data[cholesterol_column].map(cholesterol_mapping)
    
    # Create age bins for better visualization
    plot_data['age_bin'] = pd.cut(plot_data[age_column], bins=10, precision=0)
    plot_data['age_bin_mid'] = plot_data['age_bin'].apply(lambda x: int(x.mid))
    
    # Create catplot using seaborn
    # x: age bins, hue: gender, col: cholesterol level
    g = sns.catplot(
        data=plot_data,
        x='age_bin_mid',
        hue=gender_column,
        col=cholesterol_column,
        kind='count',
        height=6,
        aspect=1.2,
        palette=['#3498db', '#e74c3c']
    )
    
    g.fig.suptitle('Cholesterol Levels by Gender Over Age (Catplot)', 
                   fontsize=14, fontweight='bold', y=1.02)
    g.set_axis_labels('Age (years)', 'Count')
    g.set_xticklabels(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_catplot_smoking_balance(df, gender_column='gender', smoke_column='smoke',
                                  output_path=None):
    """
    Create catplot to examine smoking balance by gender.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    gender_column : str, optional
        Name of the gender column (default: 'gender')
    smoke_column : str, optional
        Name of the smoke column (default: 'smoke')
    output_path : str, optional
        Path to save the chart
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = 'charts/catplot_smoking_balance.png'
    
    # Prepare data
    plot_data = df[[gender_column, smoke_column]].copy().dropna()
    
    # Map values to labels
    plot_data[gender_column] = plot_data[gender_column].map({1: 'Female', 2: 'Male'})
    plot_data[smoke_column] = plot_data[smoke_column].map({0: 'Non-smoker', 1: 'Smoker'})
    
    # Create catplot
    g = sns.catplot(
        data=plot_data,
        x=gender_column,
        hue=smoke_column,
        kind='count',
        height=6,
        aspect=1.2,
        palette=['#3498db', '#e74c3c']
    )
    
    g.fig.suptitle('Smoking Balance by Gender (Catplot)', 
                   fontsize=14, fontweight='bold', y=1.02)
    g.set_axis_labels('Gender', 'Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def create_catplot_cholesterol_balance(df, gender_column='gender', 
                                      cholesterol_column='cholesterol',
                                      output_path=None):
    """
    Create catplot to examine cholesterol balance by gender.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    gender_column : str, optional
        Name of the gender column (default: 'gender')
    cholesterol_column : str, optional
        Name of the cholesterol column (default: 'cholesterol')
    output_path : str, optional
        Path to save the chart
    
    Returns:
    --------
    str : Path to the saved chart
    """
    ensure_charts_directory()
    
    if output_path is None:
        output_path = 'charts/catplot_cholesterol_balance.png'
    
    # Prepare data
    plot_data = df[[gender_column, cholesterol_column]].copy().dropna()
    
    # Map values to labels
    plot_data[gender_column] = plot_data[gender_column].map({1: 'Female', 2: 'Male'})
    cholesterol_mapping = {1: 'Normal', 2: 'Above Normal', 3: 'Well Above Normal'}
    plot_data[cholesterol_column] = plot_data[cholesterol_column].map(cholesterol_mapping)
    
    # Create catplot
    g = sns.catplot(
        data=plot_data,
        x=gender_column,
        hue=cholesterol_column,
        kind='count',
        height=6,
        aspect=1.2,
        palette=['#3498db', '#e74c3c', '#2ecc71']
    )
    
    g.fig.suptitle('Cholesterol Balance by Gender (Catplot)', 
                   fontsize=14, fontweight='bold', y=1.02)
    g.set_axis_labels('Gender', 'Count')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def generate_catplot_analysis(df, age_column='age_years', gender_column='gender',
                              smoke_column='smoke', cholesterol_column='cholesterol'):
    """
    Generate complete catplot analysis for gender correlations with smoking and cholesterol.
    
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
    cholesterol_column : str, optional
        Name of the cholesterol column (default: 'cholesterol')
    
    Returns:
    --------
    dict : Dictionary containing chart paths and balance statistics
    """
    ensure_charts_directory()
    
    print("Generating catplot analysis for gender correlations...")
    
    # Create catplots
    smoking_age_plot = create_catplot_smoking_gender_age(df, age_column, gender_column, smoke_column)
    cholesterol_age_plot = create_catplot_cholesterol_gender_age(df, age_column, gender_column, cholesterol_column)
    smoking_balance_plot = create_catplot_smoking_balance(df, gender_column, smoke_column)
    cholesterol_balance_plot = create_catplot_cholesterol_balance(df, gender_column, cholesterol_column)
    
    # Check balance
    smoking_balance = check_binary_balance(df, smoke_column, gender_column)
    # For cholesterol, we'll check balance across categories
    cholesterol_balance = check_binary_balance(df, cholesterol_column, gender_column)
    
    return {
        'smoking_age_plot': smoking_age_plot,
        'cholesterol_age_plot': cholesterol_age_plot,
        'smoking_balance_plot': smoking_balance_plot,
        'cholesterol_balance_plot': cholesterol_balance_plot,
        'smoking_balance_stats': smoking_balance,
        'cholesterol_balance_stats': cholesterol_balance
    }


def add_catplot_analysis_section_to_pdf(story, df, age_column='age_years', gender_column='gender',
                                       smoke_column='smoke', cholesterol_column='cholesterol'):
    """
    Add catplot analysis section to PDF report.
    
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
    cholesterol_column : str, optional
        Name of the cholesterol column (default: 'cholesterol')
    """
    styles = getSampleStyleSheet()
    
    # Add heading
    heading = Paragraph("Catplot Analysis: Gender Correlations with Binary Features", styles['Heading1'])
    story.append(heading)
    story.append(Spacer(1, 0.2*inch))
    
    # Add introduction
    intro_heading = Paragraph("Introduction", styles['Heading2'])
    story.append(intro_heading)
    story.append(Spacer(1, 0.1*inch))
    
    intro_text = """
    This analysis uses seaborn's catplot() function to examine quantitative variables across 
    two categorical dimensions simultaneously. We analyze gender correlations with smoking 
    and cholesterol over time (age), and examine whether the binary features are balanced 
    across gender groups.
    """
    para = Paragraph(intro_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.15*inch))
    
    # Generate analysis
    results = generate_catplot_analysis(df, age_column, gender_column, smoke_column, cholesterol_column)
    
    # Add smoking analysis
    smoking_heading = Paragraph("Smoking Patterns by Gender Over Age", styles['Heading2'])
    story.append(smoking_heading)
    story.append(Spacer(1, 0.1*inch))
    
    smoking_desc = """
    The catplot below shows smoking patterns (smoker vs non-smoker) by gender across 
    different age groups. This visualization helps identify age-related trends in smoking 
    behavior between men and women.
    """
    para = Paragraph(smoking_desc, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    add_image_to_story(story, results['smoking_age_plot'], width=5*inch)
    
    # Add cholesterol analysis
    cholesterol_heading = Paragraph("Cholesterol Levels by Gender Over Age", styles['Heading2'])
    story.append(cholesterol_heading)
    story.append(Spacer(1, 0.1*inch))
    
    cholesterol_desc = """
    The catplot below shows cholesterol levels (normal, above normal, well above normal) 
    by gender across different age groups. This visualization helps identify age-related 
    trends in cholesterol levels between men and women.
    """
    para = Paragraph(cholesterol_desc, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    add_image_to_story(story, results['cholesterol_age_plot'], width=5*inch)
    
    # Add balance analysis
    balance_heading = Paragraph("Binary Feature Balance Analysis", styles['Heading2'])
    story.append(balance_heading)
    story.append(Spacer(1, 0.1*inch))
    
    balance_intro = """
    The following analysis examines whether binary features (smoking and cholesterol categories) 
    are balanced across gender groups. Balanced data means that the proportions of different 
    categories are relatively equal across groups.
    """
    para = Paragraph(balance_intro, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.15*inch))
    
    # Smoking balance
    smoking_balance_subheading = Paragraph("Smoking Balance by Gender", styles['Heading3'])
    story.append(smoking_balance_subheading)
    story.append(Spacer(1, 0.1*inch))
    
    smoking_balance_stats = results['smoking_balance_stats']
    
    smoking_balance_text = "<b>Smoking Balance Statistics:</b><br/>"
    for group in ['Female', 'Male']:
        if group in smoking_balance_stats:
            stats = smoking_balance_stats[group]
            smoking_balance_text += f"""
            <b>{group}:</b><br/>
            Total: {stats['total']:,}<br/>
            """
            for value, count in stats['value_counts'].items():
                prop = stats['proportions'][value]
                smoking_balance_text += f"  {value}: {count:,} ({prop*100:.2f}%)<br/>"
            smoking_balance_text += "<br/>"
    
    if 'overall' in smoking_balance_stats:
        overall = smoking_balance_stats['overall']
        is_balanced = "Yes" if overall['is_balanced'] else "No"
        smoking_balance_text += f"""
        <b>Overall Balance:</b><br/>
        Balance Ratio: {overall['balance_ratio']:.3f}<br/>
        Is Balanced: {is_balanced}<br/>
        """
    
    para = Paragraph(smoking_balance_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    add_image_to_story(story, results['smoking_balance_plot'], width=5*inch)
    
    # Cholesterol balance
    cholesterol_balance_subheading = Paragraph("Cholesterol Balance by Gender", styles['Heading3'])
    story.append(cholesterol_balance_subheading)
    story.append(Spacer(1, 0.1*inch))
    
    cholesterol_balance_stats = results['cholesterol_balance_stats']
    
    cholesterol_balance_text = "<b>Cholesterol Balance Statistics:</b><br/>"
    for group in ['Female', 'Male']:
        if group in cholesterol_balance_stats:
            stats = cholesterol_balance_stats[group]
            cholesterol_balance_text += f"""
            <b>{group}:</b><br/>
            Total: {stats['total']:,}<br/>
            """
            for value, count in stats['value_counts'].items():
                prop = stats['proportions'][value]
                cholesterol_balance_text += f"  {value}: {count:,} ({prop*100:.2f}%)<br/>"
            cholesterol_balance_text += "<br/>"
    
    if 'overall' in cholesterol_balance_stats:
        overall = cholesterol_balance_stats['overall']
        is_balanced = "Yes" if overall['is_balanced'] else "No"
        cholesterol_balance_text += f"""
        <b>Overall Balance:</b><br/>
        Balance Ratio: {overall['balance_ratio']:.3f}<br/>
        Is Balanced: {is_balanced}<br/>
        """
    
    para = Paragraph(cholesterol_balance_text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))
    
    add_image_to_story(story, results['cholesterol_balance_plot'], width=5*inch)
    