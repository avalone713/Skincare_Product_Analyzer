import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re

# --- Load and Process Main Product Data ---
print("Loading main product data...")
df_final = pd.read_csv(".../data/processed/final_merged_products.csv", low_memory=False)

print(df_final.head())
print(df_final.info())
print(df_final.describe(include='all'))

# --- Category Processing ---
print("\nProcessing categories...")
df_final = df_final.rename(columns={
    'category': 'category_2',
    'Label': 'category_1'
})

categories = []
mismatch_count = 0

for idx, row in df_final.iterrows():
    cat1 = str(row['category_1']).strip().lower()
    cat2_raw = row['category_2']
    
    if pd.isna(cat2_raw):
        categories.append(cat1)
    else:
        cat2 = str(cat2_raw).strip().lower()
        if cat1 == cat2 or cat1 + 's' == cat2:
            categories.append(cat1)
        else:
            #print(f"[Index {idx}] Mismatch: {row['Name']} | category_1: {cat1} | category_2: {cat2}")
            categories.append(np.nan)
            mismatch_count += 1

# Assign the new 'category' column
df_final['category'] = categories

# Step 4: Summary output
print(f"\nTotal mismatches (NaN in 'category'): {df_final['category'].isna().sum()} (Counted: {mismatch_count})")

# Apply category fixes
print("\nApplying category fixes...")
# Rule 1: If cat1 is 'sun protect'
sun_protect_fix = df_final['category_1'].str.strip().str.lower() == 'sun protect'

# Rule 2: cat1 is 'eye cream' and cat2 is 'eye care'
eye_fix = (
    df_final['category_1'].str.strip().str.lower() == 'eye cream'
) & (
    df_final['category_2'].str.strip().str.lower() == 'eye care'
)

# Rule 3+: Index-specific fixes
cat1s_indices = [363,384,422,427,436,441,446,448,456,481,482,510,568,741,779,782,784,792,804,900,908,922,924,928,996,1011,1036,1038,1069,1070]
cat2s_indices = [714, 845]
cat2s_plural_indices = [336,351,378,387,417,511,522,538,558,725,813,826]
toner_indices = [452, 576]
treatment_indices = [513, 524]

# Apply fixes
for idx in df_final[df_final['category'].isna()].index:
    cat1 = str(df_final.at[idx, 'category_1']).strip().lower()
    cat2 = str(df_final.at[idx, 'category_2']).strip().lower() if pd.notna(df_final.at[idx, 'category_2']) else None
    
    if idx in cat1s_indices:
        df_final.at[idx, 'category'] = cat1
    elif idx in cat2s_indices:
        df_final.at[idx, 'category'] = cat2
    elif idx in cat2s_plural_indices:
        df_final.at[idx, 'category'] = cat2[:-1] if cat2.endswith('s') else cat2
    elif idx in toner_indices:
        df_final.at[idx, 'category'] = 'treatment'
    elif idx in treatment_indices:
        df_final.at[idx, 'category'] = 'treatment'
    elif cat1 == 'sun protect':
        df_final.at[idx, 'category'] = cat1
    elif cat1 == 'eye cream' and cat2 == 'eye care':
        df_final.at[idx, 'category'] = cat2
    elif df_final.at[idx, 'category'] == 'cleansers' or 'treatments':
        cat = str(df_final.at[idx, 'category']).strip().lower()
        df_final.at[idx, 'category'] = cat[:-1]

# Manual overrides for specific cases
df_final.loc[352, 'category'] = df_final.loc[352, 'category_1'].lower()
df_final.loc[495, 'category'] = 'treatment'
df_final.loc[1188, 'category'] = 'eye care'

# Replace all 'eye cream' with 'eye care'
mask_eye_cream = df_final['category'].str.lower() == 'eye cream'
df_final.loc[mask_eye_cream, 'category'] = 'eye care'

print("\nCategory distribution after fixes:")
category_counts = df_final['category'].value_counts(dropna=False).sort_index()
print(category_counts)

# --- Ingredient Processing ---
print("\nProcessing ingredients...")

# First, let's find all ingredients containing "1,"
print("\n--- Ingredients containing '1,' ---")
all_ingredients = df_final['Ingredients'].dropna().astype(str)
ingredients_with_1 = []
for ing_string in all_ingredients:
    ingredients = [ing.strip() for ing in ing_string.split(',')]
    for ing in ingredients:
        if '1,' in ing:
            ingredients_with_1.append(ing.strip())
print("\nUnique ingredients containing '1,':")
for ing in sorted(set(ingredients_with_1)):
    print(f"- {ing}")

def split_ingredients(ingredients_str):
    """
    Split ingredients string while preserving compound names that contain commas.
    For example, '1,2-Hexanediol' should remain as one ingredient.
    Returns a semicolon-separated string of ingredients.
    """
    if pd.isna(ingredients_str) or not ingredients_str.strip():
        return ""
    
    # First, replace commas in compound names with a temporary marker
    # Look for patterns like "1,2-Hexanediol" or "C10-30 Alkyl Acrylate"
    # Replace commas with periods in compound names
    ingredients_str = re.sub(r'(\d+,\d+-\w+)', lambda m: m.group(1).replace(',', '.'), ingredients_str)
    
    # Split by comma
    ingredients = [ing.strip() for ing in ingredients_str.split(',')]
    
    # Restore commas in compound names (convert periods back to commas)
    ingredients = [ing.replace('.', ',') for ing in ingredients]
    
    # Filter out water and its variations
    water_variations = ['water', 'aqua', 'eau', 'water/eau', 'purified water', 'distilled water']
    ingredients = [ing for ing in ingredients if ing.lower() not in water_variations]
    
    # Filter out empty strings and join with semicolons
    return ';'.join([ing for ing in ingredients if ing])

# Process ingredients
df_final['processed_ingredients'] = df_final['Ingredients'].apply(split_ingredients)
df_final['num_ingredients_approx'] = df_final['processed_ingredients'].apply(lambda x: len(x.split(';')) if x else 0)

print("\n--- Basic Statistics on Number of Ingredients per Product ---")
print(df_final['num_ingredients_approx'].describe())

# Most Common Raw Ingredient Tokens
print("\n--- Most Common Raw Ingredient Tokens (Top 50, Lowercased) ---")
all_raw_tokens = []
for ing_string in df_final['processed_ingredients'].dropna():
    tokens = [token.strip().lower() for token in ing_string.split(';')]
    all_raw_tokens.extend(tokens)
    token_counts = Counter(all_raw_tokens)
    
print("Top 50 most common raw ingredient tokens (lowercase) and their counts:")
for token, count in token_counts.most_common(50):
    print(f"- \"{token}\": {count}")

# Clean up temporary columns
df_final.drop(columns=['num_ingredients_approx'], inplace=True)

# Only use non-null values for binning
loves_nonnull = df_final['n_of_loves'].dropna()
# Use qcut to create 5 bins (quintiles), adjust number of bins as needed
df_final['n_of_loves_bin'], bin_edges = pd.qcut(
    df_final['n_of_loves'], 
    q=5, 
    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    retbins=True,
    duplicates='drop'
)

def k_format(n):
    n = int(round(n, -3))
    if n >= 1000:
        return f"{int(n/1000)}k"
    else:
        return str(int(n))
    
rounded_edges = np.round(bin_edges, -3)

bin_ranges = {label: f"{k_format(rounded_edges[i])} - {k_format(rounded_edges[i+1])}" 
              for i, label in enumerate(['Very Low', 'Low', 'Medium', 'High', 'Very High']) 
              if i < len(rounded_edges) - 1}

# After bin_ranges is created
label_map = {label: f"{label} ({rng})" for label, rng in bin_ranges.items()}
df_final['n_of_loves_bin'] = df_final['n_of_loves_bin'].astype(str).map(label_map).fillna(df_final['n_of_loves_bin'].astype(str))

# Save the processed data
print("\nSaving processed data...")
df_final.to_csv('.../data/processed/final_products.csv', index=False)

print("\nData preparation complete!")

