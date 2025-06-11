import pandas as pd
import numpy as np
import re

# Load the data
print("Loading data...")
df_paula = pd.read_csv('.../data/raw/Paula_embedding_SUMLIST_before_422.csv', low_memory=False)
df_final = pd.read_csv('.../data/processed/final_products.csv', low_memory=False)
df_skincare = pd.read_csv('.../data/processed/skincare_df_cleaned.csv', low_memory=False)

print("\n--- Original DataFrames Info ---")
print("\nFinal Products DataFrame:")
print(df_final.info())
print("\nSkincare DataFrame:")
print(df_skincare.info())

# Create match keys for merging
df_skincare['match_key'] = df_skincare['brand_name_std'] + "_" + df_skincare['product_name_std']

# Process Paula's Choice data
print("\nProcessing Paula's Choice data...")

print(df_paula.head())
print(df_paula.info())
print(df_paula.describe(include='all'))

# Convert rating to numerical values
def convert_rating(rating):
    if pd.isna(rating):
        return np.nan
    if isinstance(rating, (int, float)):
        return float(rating)
    rating = str(rating).upper()
    rating_map = {
        'BEST': 5.0,
        'GOOD': 4.0,
        'AVERAGE': 3.0,
        'BAD': 2.0,
        'WORST': 1.0,
        'NOT RATED': np.nan
    }
    return rating_map.get(rating, np.nan)

# Process benefits column
def process_benefits(benefits_str):
    if pd.isna(benefits_str):
        return []
    # Split by ';;' and clean each benefit
    benefits = [benefit.strip() for benefit in str(benefits_str).split(';;')]
    return [benefit for benefit in benefits if benefit]

# Create a dictionary for quick ingredient lookup
print("Creating ingredient lookup dictionary...")
ingredient_dict = {}
for _, row in df_paula.iterrows():
    ingredient_name = str(row['ingredient_name']).lower().strip()
    ingredient_dict[ingredient_name] = {
        'description': row['description'] if pd.notna(row['description']) else '',
        'functions': row['functions'] if pd.notna(row['functions']) else '',
        'benefits': process_benefits(row['benefits']),
        'category': row['categories'] if pd.notna(row['categories']) else '',
        'rating': convert_rating(row['rating'])
    }

# Process ingredients for each product
print("\nProcessing product ingredients...")
def get_ingredient_details(ingredients_str):
    if pd.isna(ingredients_str):
        return []
    
    # Split ingredients by semicolon
    ingredients = [ing.strip().lower() for ing in ingredients_str.split(';')]
    
    # Look up each ingredient in the dictionary
    ingredient_details = []
    for ing in ingredients:
        if ing in ingredient_dict:
            ingredient_details.append({
                'name': ing,
                **ingredient_dict[ing]
            })
    
    return ingredient_details

# Add ingredient details to df_final
print("Adding ingredient details to products...")
df_final['paula_ingredient_details'] = df_final['processed_ingredients'].apply(get_ingredient_details)

# Print some statistics
print("\n--- Statistics ---")
total_products = len(df_final)
products_with_ingredients = df_final['paula_ingredient_details'].apply(len).gt(0).sum()
print(f"Total products: {total_products}")
print(f"Products with matched ingredients: {products_with_ingredients}")
print(f"Percentage of products with matched ingredients: {(products_with_ingredients/total_products)*100:.2f}%")

# Count total unique ingredients matched
all_matched_ingredients = set()
for details in df_final['paula_ingredient_details']:
    all_matched_ingredients.update(d['name'] for d in details)
print(f"\nTotal unique ingredients matched with Paula's Choice data: {len(all_matched_ingredients)}")

# Remove all columns that start with 'category_'
df_final = df_final.loc[:, ~df_final.columns.str.startswith('category_')]

# Save the processed data
print("\nSaving processed data...")
df_final.to_csv('/Users/alexvalone/Desktop/Data_Science/DS_Q3/Visualization/skincare_project/data/final_products_ingredients.csv', index=False)

print("\nData preparation complete!")
