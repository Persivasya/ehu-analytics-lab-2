import pandas as pd
from reportlab.lib.pagesizes import elevenSeventeen, letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate
from boxplot_functions import add_boxplot_section_to_pdf
from generate_pdf import generate_statistics_report
from histogram_functions import add_histogram_section_to_pdf
from scatterplot_functions import add_scatterplot_section_to_pdf
from violinplot_functions import add_violinplot_section_to_pdf

# Load the dataset
df = pd.read_csv('mlbootcamp5.csv', sep=';')

# 1. Create age in years feature (convert from days to years, int type)
df['age_years'] = (df['age'] / 365.25).astype(int)

# 2. Create BMI feature (weight in kg / (height in m)^2)
# Height is in cm, so convert to meters by dividing by 100
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

# 3. Get statistics information about the dataset
df_stats = df.describe()

# 4. Get information about each column type
column_info = pd.DataFrame({
    'column': df.columns,
    'dtype': df.dtypes,
    'dtype_name': df.dtypes.astype(str),
    'non_null': df.count()
})
column_info.set_index('column', inplace=True)

# Create PDF report
doc = SimpleDocTemplate("statistics_report.pdf", pagesize=elevenSeventeen)
styles = getSampleStyleSheet()
story = []

# Generate PDF report
generate_statistics_report(df, df_stats, column_info, story)
add_histogram_section_to_pdf(story, df)
add_boxplot_section_to_pdf(story, df)
add_scatterplot_section_to_pdf(story, df)
add_violinplot_section_to_pdf(story, df)

# Build PDF
doc.build(story)
print(f"PDF report generated: statistics_report.pdf")
