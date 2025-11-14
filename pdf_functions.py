import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib import colors
from reportlab.lib.units import inch


def add_title(story, title_text):
    """Add a title to the PDF story."""
    styles = getSampleStyleSheet()
    title = Paragraph(title_text, styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.2*inch))


def add_heading(story, heading_text):
    """Add a heading to the PDF story."""
    styles = getSampleStyleSheet()
    heading = Paragraph(heading_text, styles['Heading2'])
    story.append(heading)
    story.append(Spacer(1, 0.1*inch))


def add_statistics_table(story, df_stats):
    """Add statistics table to the PDF story."""
    styles = getSampleStyleSheet()
    
    # Convert DataFrame to list of lists for Table
    data = [['Statistic'] + list(df_stats.columns)]
    
    for index in df_stats.index:
        row = [str(index)]
        for col in df_stats.columns:
            value = df_stats.loc[index, col]
            if isinstance(value, float):
                row.append(f"{value:.2f}")
            else:
                row.append(str(value))
        data.append(row)
    
    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.2*inch))


def add_column_info_table(story, df_info):
    """Add column information table to the PDF story."""
    styles = getSampleStyleSheet()
    
    # Convert DataFrame info to list of lists for Table
    data = [['Column', 'Type']]
    
    for col in df_info.index:
        dtype_str = str(df_info.loc[col, 'dtype'])
        dtype_name = str(df_info.loc[col, 'dtype_name'])
        non_null = str(df_info.loc[col, 'non_null'])
        data.append([col, dtype_name, non_null, dtype_str])
    
    # Create table
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))
    
    story.append(table)
    story.append(Spacer(1, 0.2*inch))


def add_text(story, text):
    """Add plain text paragraph to the PDF story."""
    styles = getSampleStyleSheet()
    para = Paragraph(text, styles['Normal'])
    story.append(para)
    story.append(Spacer(1, 0.1*inch))

def add_image_to_story(story, image_path, width=5*inch, height=4*inch):
    """
    Add an image to the PDF story.
    
    Parameters:
    -----------
    story : list
        The PDF story list
    image_path : str
        Path to the image file
    width : float, optional
        Width of the image in inches (default: 5 inches to fit within page margins)
    height : float, optional
        Height of the image in inches (default: None, maintains aspect ratio)
    """
    if os.path.exists(image_path):
        # Ensure width doesn't exceed page width (letter size is 8.5 inches, with margins ~6.5 inches usable)
        # Use 5 inches to be safe
        max_width = 5 * inch
        if width > max_width:
            width = max_width
        img = Image(image_path, width=width, height=height)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    else:
        print(f"Warning: Image file {image_path} not found.")
