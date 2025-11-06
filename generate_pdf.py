from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate
from pdf_functions import (
    add_title, add_heading, add_statistics_table, 
    add_column_info_table, add_text
)


def generate_statistics_report(df, df_stats, column_info, story, output_filename="statistics_report.pdf"):
    """
    Generate a PDF report with dataset statistics and column information.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The processed dataframe
    df_stats : pandas.DataFrame
        Statistical summary of the dataset
    column_info : pandas.DataFrame
        Column information (types, dtypes, non-null counts)
    output_filename : str, optional
        Name of the output PDF file (default: "statistics_report.pdf")
    """
    # Add title
    add_title(story, "Dataset Statistics Report")
    
    # Add dataset overview
    add_heading(story, "Dataset Overview")
    add_text(story, f"Total number of rows: {len(df)}")
    add_text(story, f"Total number of columns: {len(df.columns)}")
    
    # Add statistics table
    add_heading(story, "Statistical Summary")
    add_statistics_table(story, df_stats)
    
    # Add column information
    add_heading(story, "Column Information")
    add_column_info_table(story, column_info)
    
    # Add BMI information
    add_heading(story, "BMI Analysis")
    bmi_normal = df[(df['BMI'] >= 18.5) & (df['BMI'] <= 25)]
    add_text(story, f"Number of records with normal BMI (18.5-25): {len(bmi_normal)}")
    add_text(story, f"Percentage of records with normal BMI: {len(bmi_normal)/len(df)*100:.2f}%")
    add_text(story, f"BMI statistics:")
    bmi_stats = df['BMI'].describe()
    for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        add_text(story, f"  {stat}: {bmi_stats[stat]:.2f}")
    

