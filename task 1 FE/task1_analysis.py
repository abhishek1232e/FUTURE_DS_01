    # ============================================
# COMPLETE SALES ANALYSIS - TASK 1 SOLUTION
# ============================================

print("🔍 FUTURE INTERNS - TASK 1: SALES PERFORMANCE ANALYTICS")
print("=" * 60)
print()

# Step 1: Import libraries
print("Loading libraries...")
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    print("✓ Libraries loaded successfully!")
except:
    print("✗ Error: Please install libraries first!")
    print("Open Command Prompt and type:")
    print("py -m pip install pandas matplotlib")
    input("Press Enter to exit...")
    exit()

print()
print("=" * 60)
print("DATA LOADING")
print("=" * 60)
print()

# Step 2: Load your data
print("IMPORTANT: Put your CSV file in SAME FOLDER as this program")
print("Common dataset names: superstore.csv, retail.csv, sales_data.csv")

# List available CSV files
import os
csv_files = [f for f in os.listdir('.') if f.lower().endswith('.csv')]

if csv_files:
    print("\nCSV files found in this folder:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f}")
else:
    print("\nNo CSV files found in this folder.")
    print("Please put your CSV file here and run again.")
    input("Press Enter to exit...")
    exit()

print()
# COMMENT OUT the input and use sample_data.csv directly
print("Using sample_data.csv for GitHub submission...")
file_name = "sample_data.csv"  # Always use sample file
print(f"\nLoading {file_name}...")

try:
    df = pd.read_csv('sample_data.csv')
    print(f"✓ Success! Loaded {len(df)} rows and {len(df.columns)} columns")
except Exception as e:
    print(f"✗ Error loading file: {e}")
    print("Please check filename and try again.")
    input("Press Enter to exit...")
    exit()

print()
print("=" * 60)
print("DATA PREVIEW")
print("=" * 60)
print()

# Show data structure
print("First 5 rows:")
print(df.head())
print()

print("Column names in your dataset:")
print("-" * 40)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col} (Type: {df[col].dtype})")
print()

print("=" * 60)
print("DATA PREPARATION")
print("=" * 60)
print()

# Identify key columns automatically
print("Automatically detecting important columns...")

# Find potential date column
date_cols = [col for col in df.columns if any(word in col.lower() for word in ['date', 'time', 'year', 'month', 'day'])]
date_col = date_cols[0] if date_cols else None

# Find potential product column
product_cols = [col for col in df.columns if any(word in col.lower() for word in ['product', 'item', 'description', 'name'])]
product_col = product_cols[0] if product_cols else df.columns[0]

# Find potential quantity column
qty_cols = [col for col in df.columns if any(word in col.lower() for word in ['qty', 'quantity', 'unit', 'count'])]
qty_col = qty_cols[0] if qty_cols else None

# Find potential price column
price_cols = [col for col in df.columns if any(word in col.lower() for word in ['price', 'cost', 'amount', 'revenue', 'sales'])]
price_col = price_cols[0] if price_cols else None

# Find potential category column
category_cols = [col for col in df.columns if any(word in col.lower() for word in ['category', 'type', 'class', 'segment'])]
category_col = category_cols[0] if category_cols else None

# Display what we found
print(f"✓ Date Column: {date_col if date_col else 'Not found'}")
print(f"✓ Product Column: {product_col}")
print(f"✓ Quantity Column: {qty_col if qty_col else 'Not found'}")
print(f"✓ Price Column: {price_col if price_col else 'Not found'}")
print(f"✓ Category Column: {category_col if category_col else 'Not found'}")
print()

# Create Revenue column if possible
if qty_col and price_col:
    df['Revenue'] = df[qty_col] * df[price_col]
    print(f"✓ Created 'Revenue' column = {qty_col} × {price_col}")
elif price_col:
    df['Revenue'] = df[price_col]
    print(f"✓ Created 'Revenue' column using {price_col}")
else:
    # Find any numeric column to use as revenue
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df['Revenue'] = df[numeric_cols[0]]
        print(f"✓ Created 'Revenue' column using {numeric_cols[0]}")
    else:
        df['Revenue'] = 100  # Default value
        print("⚠ Created 'Revenue' column with default value 100")

print()
print("=" * 60)
print("ANALYSIS 1: TOP PRODUCTS BY REVENUE")
print("=" * 60)
print()

# Find top products
top_n = 10
if len(df[product_col].unique()) > 1:
    top_products = df.groupby(product_col)['Revenue'].sum().sort_values(ascending=False).head(top_n)
    
    print(f"TOP {top_n} PRODUCTS:")
    print("-" * 50)
    for i, (product, revenue) in enumerate(top_products.items(), 1):
        print(f"{i:2}. {product[:40]:40} ${revenue:,.2f}")
    
    # Create chart
    plt.figure(figsize=(12, 6))
    top_products.plot(kind='bar', color='steelblue')
    plt.title(f'Top {top_n} Products by Revenue', fontsize=16)
    plt.xlabel('Product')
    plt.ylabel('Revenue ($)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('top_products.png', dpi=100)
    print(f"\n✓ Chart saved as 'top_products.png'")
else:
    print("Only one product found in data.")

print()
print("=" * 60)
print("ANALYSIS 2: REVENUE TREND OVER TIME")
print("=" * 60)
print()

if date_col:
    try:
        # Convert to datetime
        df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Date'])
        
        # Group by month
        df['Month'] = df['Date'].dt.to_period('M')
        monthly_revenue = df.groupby('Month')['Revenue'].sum()
        
        # Convert for plotting
        monthly_revenue.index = monthly_revenue.index.to_timestamp()
        
        print("MONTHLY REVENUE TREND:")
        print("-" * 40)
        for month, revenue in monthly_revenue.items():
            print(f"{month.strftime('%b %Y')}: ${revenue:,.2f}")
        
        # Create chart
        plt.figure(figsize=(14, 6))
        plt.plot(monthly_revenue.index, monthly_revenue.values, 
                marker='o', linewidth=2, color='green', markersize=8)
        plt.title('Monthly Revenue Trend', fontsize=16)
        plt.xlabel('Month')
        plt.ylabel('Revenue ($)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('revenue_trend.png', dpi=100)
        print(f"\n✓ Chart saved as 'revenue_trend.png'")
        
        # Calculate growth
        if len(monthly_revenue) > 1:
            growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[0]) / monthly_revenue.iloc[0]) * 100
            print(f"📈 Overall Growth: {growth:.1f}%")
            
    except Exception as e:
        print(f"Could not analyze time trend: {e}")
else:
    print("No date column found for time analysis.")

print()
print("=" * 60)
print("ANALYSIS 3: CATEGORY PERFORMANCE")
print("=" * 60)
print()

if category_col:
    category_revenue = df.groupby(category_col)['Revenue'].sum().sort_values(ascending=False)
    
    print("REVENUE BY CATEGORY:")
    print("-" * 40)
    total_revenue = category_revenue.sum()
    for category, revenue in category_revenue.items():
        percentage = (revenue / total_revenue) * 100
        print(f"{category:20} ${revenue:10,.2f} ({percentage:.1f}%)")
    
    # Create chart
    plt.figure(figsize=(10, 8))
    plt.pie(category_revenue.values, labels=category_revenue.index, 
            autopct='%1.1f%%', startangle=90)
    plt.title('Revenue Distribution by Category', fontsize=16)
    plt.tight_layout()
    plt.savefig('category_pie.png', dpi=100)
    print(f"\n✓ Chart saved as 'category_pie.png'")
    
    # Find top category
    top_category = category_revenue.index[0]
    top_percentage = (category_revenue.iloc[0] / total_revenue) * 100
    print(f"\n🏆 Top Category: {top_category} ({top_percentage:.1f}% of total)")
    
else:
    print("No category column found.")
    # Try to create simple groups from other columns
    for col in ['Country', 'Region', 'Segment']:
        if col in df.columns:
            print(f"\nAnalyzing by {col} instead...")
            group_revenue = df.groupby(col)['Revenue'].sum().sort_values(ascending=False)
            print(f"Top 3 by {col}:")
            for i, (group, revenue) in enumerate(group_revenue.head(3).items(), 1):
                print(f"  {i}. {group}: ${revenue:,.2f}")
            break

print()
print("=" * 60)
print("ANALYSIS 4: SUMMARY STATISTICS")
print("=" * 60)
print()

total_revenue = df['Revenue'].sum()
avg_revenue = df['Revenue'].mean()
max_revenue = df['Revenue'].max()
min_revenue = df['Revenue'].min()

print(f"💰 Total Revenue: ${total_revenue:,.2f}")
print(f"📊 Average Transaction: ${avg_revenue:,.2f}")
print(f"⬆️  Maximum: ${max_revenue:,.2f}")
print(f"⬇️  Minimum: ${min_revenue:,.2f}")
print(f"📈 Total Transactions: {len(df):,}")

print()
print("=" * 60)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("=" * 60)
print()

# Generate insights
insights = []

# Insight 1: Top product
if 'top_products' in locals() and len(top_products) > 0:
    top_product = top_products.index[0]
    top_revenue = top_products.iloc[0]
    top_percent = (top_revenue / total_revenue) * 100
    insights.append(f"1. **Focus on Top Product**: '{top_product}' generates ${top_revenue:,.2f} ({top_percent:.1f}% of total revenue). Consider increasing stock and marketing for this product.")

# Insight 2: Time trend
if 'monthly_revenue' in locals() and len(monthly_revenue) > 1:
    best_month = monthly_revenue.idxmax().strftime('%B %Y')
    worst_month = monthly_revenue.idxmin().strftime('%B %Y')
    insights.append(f"2. **Seasonal Pattern**: Highest revenue in {best_month}, lowest in {worst_month}. Plan inventory and promotions accordingly.")

# Insight 3: Category
if 'category_revenue' in locals() and len(category_revenue) > 1:
    top_cat = category_revenue.index[0]
    bottom_cat = category_revenue.index[-1]
    insights.append(f"3. **Category Strategy**: '{top_cat}' is the strongest category. Consider why '{bottom_cat}' underperforms - improve or phase out.")

# Insight 4: General
insights.append("4. **Customer Focus**: Analyze customer segments to identify high-value customers for loyalty programs.")
insights.append("5. **Growth Opportunity**: Consider expanding product lines in top-performing categories.")

# Print insights
for insight in insights:
    print(f"• {insight}")

print()
print("=" * 60)
print("FINAL REPORT GENERATION")
print("=" * 60)
print()

# Create comprehensive report
report = f"""
{'=' * 60}
FUTURE INTERNS - TASK 1: BUSINESS SALES PERFORMANCE ANALYTICS
{'=' * 60}

EXECUTIVE SUMMARY
-----------------
• Dataset: {file_name}
• Analysis Period: {df[date_col].min() if date_col else 'N/A'} to {df[date_col].max() if date_col else 'N/A'}
• Total Revenue Analyzed: ${total_revenue:,.2f}
• Total Transactions: {len(df):,}

KEY FINDINGS
------------
1. REVENUE OVERVIEW:
   • Total Revenue: ${total_revenue:,.2f}
   • Average per Transaction: ${avg_revenue:,.2f}
   • Revenue Range: ${min_revenue:,.2f} to ${max_revenue:,.2f}

2. TOP PERFORMERS:
"""

if 'top_products' in locals():
    report += "   Top 5 Products by Revenue:\n"
    for i, (product, revenue) in enumerate(top_products.head(5).items(), 1):
        report += f"   {i}. {product}: ${revenue:,.2f}\n"

if 'category_revenue' in locals():
    report += "\n3. CATEGORY PERFORMANCE:\n"
    for i, (category, revenue) in enumerate(category_revenue.head(3).items(), 1):
        percent = (revenue / total_revenue) * 100
        report += f"   {i}. {category}: ${revenue:,.2f} ({percent:.1f}%)\n"

if 'monthly_revenue' in locals():
    report += "\n4. TIME TREND:\n"
    if len(monthly_revenue) > 1:
        growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[0]) / monthly_revenue.iloc[0]) * 100
        report += f"   • Overall Growth Rate: {growth:.1f}%\n"
        report += f"   • Peak Month: {monthly_revenue.idxmax().strftime('%B %Y')}\n"
        report += f"   • Lowest Month: {monthly_revenue.idxmin().strftime('%B %Y')}\n"

report += f"""
ACTIONABLE RECOMMENDATIONS
--------------------------
"""

for insight in insights:
    report += f"• {insight}\n"

report += f"""
DELIVERABLES CREATED
--------------------
1. top_products.png - Bar chart of top products
2. revenue_trend.png - Line chart of revenue over time
3. category_pie.png - Pie chart of category distribution
4. This comprehensive report

CONCLUSION
----------
This analysis provides clear insights into sales performance, 
highlighting top products, seasonal trends, and category performance. 
The recommendations offer actionable steps for business growth.

{'=' * 60}
Analysis completed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 60}
"""

# Save report
with open('business_sales_analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

# Also save insights separately
with open('key_insights.txt', 'w', encoding='utf-8') as f:
    f.write("KEY BUSINESS INSIGHTS\n")
    f.write("=" * 40 + "\n\n")
    for insight in insights:
        f.write(f"• {insight}\n")

print("✓ Final report saved as 'business_sales_analysis_report.txt'")
print("✓ Key insights saved as 'key_insights.txt'")
print("✓ All charts saved as PNG files")
print()

print("=" * 60)
print("🎉 TASK 1 COMPLETED SUCCESSFULLY!")
print("=" * 60)
print()
print("📤 WHAT TO SUBMIT FOR YOUR INTERNSHIP:")
print("1. business_sales_analysis_report.txt")
print("2. key_insights.txt")
print("3. top_products.png")
print("4. revenue_trend.png (if created)")
print("5. category_pie.png (if created)")
print("6. Screenshot of this program running")
print()
print("💡 TIP: Create a folder called 'Task1_Submission'")
print("       and put all these files inside.")
print()
print("Good luck with your internship! 🚀")
input("Press Enter to exit...")
