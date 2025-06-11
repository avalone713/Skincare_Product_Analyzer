import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
import argparse
from Levenshtein import ratio

def standardize_name(name):
    """
    Standardize a name (can handle both Series and individual strings)
    """
    if pd.isna(name):
        return name
    
    # Convert to string and lowercase
    name = str(name).lower()
    
    # Remove content within parentheses and the parentheses themselves
    name = re.sub(r"\(.*?\)", "", name)
    
    # Replace characters like +, ™, ®, etc., with underscore or remove them
    name = re.sub(r'[™®©&+/()%-]', '_', name)
    
    # Replace multiple spaces or underscores with a single underscore
    name = re.sub(r'\s+', '_', name)
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    return name

def standardize_brand_name(brand):
    """
    Standardize brand names to handle special characters and common variations.
    """
    if pd.isna(brand):
        return brand
    
    brand = str(brand).strip()
    
    # Common brand name fixes
    brand_fixes = {
        'Est√©e Lauder': 'ESTÉE LAUDER',
        'Dr Roebuck\'s': 'DR ROEBUCK\'S',
        'Dr. Roebuck\'s': 'DR ROEBUCK\'S',
        'Dr. Roebucks': 'DR ROEBUCK\'S',
        'Dr Roebucks': 'DR ROEBUCK\'S'
    }
    
    # Apply fixes if the brand matches any of the known variations
    for wrong, correct in brand_fixes.items():
        if brand.lower() == wrong.lower():
            return correct
    
    return brand

def process_category_files():
    """Process category files and create product to category mapping"""
    category_product_data = []
    category_files_info = {
        "Moisturizers": "moisturizers.csv",
        "Eye Care": "eyecare.csv",
        "Treatments": "treatments.csv"
    }

    base_path = "/Users/alexvalone/Desktop/Data_Science/DS_Q3/Visualization/skincare_project/data/seph_skin_products"
    
    for category, file_path in category_files_info.items():
        full_path = f"{base_path}/{file_path}"
        df_cat_file = pd.read_csv(full_path)
        # Standardize brand and name to create match_key
        df_cat_file['brand_name_std'] = df_cat_file['brand'].apply(standardize_name)
        df_cat_file['product_name_std'] = df_cat_file['name'].apply(standardize_name)
        df_cat_file['match_key'] = df_cat_file['brand_name_std'] + "_" + df_cat_file['product_name_std']
        
        df_cat_file['category'] = category
        category_product_data.append(df_cat_file)

    df_all_cat_strings = pd.concat(category_product_data, ignore_index=True)
    return df_all_cat_strings.drop_duplicates(subset=['match_key'], keep='first').set_index('match_key')['category'].to_dict()

def find_potential_matches(df_base, df_sephora, name_threshold=0.8, brand_threshold=0.85):
    """
    Find potential matches between base and Sephora DataFrames based on similarity thresholds.
    """
    unmatched_base = df_base[~df_base['match_key'].isin(df_sephora['match_key'])]
    unmatched_sephora = df_sephora[~df_sephora['match_key'].isin(df_base['match_key'])]
    
    print(f"\nBefore analysis:")
    print(f"Unmatched products in base DataFrame: {len(unmatched_base)}")
    print(f"Unmatched products in Sephora DataFrame: {len(unmatched_sephora)}")
    
    potential_matches = []
    for _, base_row in unmatched_base.iterrows():
        base_name = base_row['product_name_std']
        base_brand = base_row['brand_name_std']
        
        for _, sephora_row in unmatched_sephora.iterrows():
            sephora_name = sephora_row['product_name_std']
            sephora_brand = sephora_row['brand_name_std']
            
            name_similarity = ratio(str(base_name).lower(), str(sephora_name).lower())
            brand_similarity = ratio(str(base_brand).lower(), str(sephora_brand).lower())
            
            if name_similarity > name_threshold or brand_similarity > brand_threshold:
                potential_matches.append({
                    'base_name': base_row['Name'],
                    'base_brand': base_row['Brand'],
                    'base_name_std': base_name,
                    'base_brand_std': base_brand,
                    'sephora_name': sephora_row['name'],
                    'sephora_brand': sephora_row['brand'],
                    'sephora_name_std': sephora_name,
                    'sephora_brand_std': sephora_brand,
                    'name_similarity': name_similarity,
                    'brand_similarity': brand_similarity,
                    'should_merge': name_similarity >= name_threshold,
                    'match_type': 'exact' if name_similarity == 1.0 else 'high_similarity'
                })
    
    return potential_matches, unmatched_base, unmatched_sephora

def analyze_similarity_distribution(potential_matches_df):
    """
    Analyze and print statistics about the similarity distribution.
    """
    print("\nSimilarity Distribution Analysis:")
    
    # Name similarity distribution
    name_sim_ranges = [(0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
    print("\nName Similarity Distribution:")
    for lower, upper in name_sim_ranges:
        count = len(potential_matches_df[
            (potential_matches_df['name_similarity'] >= lower) & 
            (potential_matches_df['name_similarity'] < upper)
        ])
        print(f"{lower:.2f}-{upper:.2f}: {count} matches")
    
    # Brand similarity distribution
    brand_sim_ranges = [(0.8, 0.85), (0.85, 0.9), (0.9, 0.95), (0.95, 1.0)]
    print("\nBrand Similarity Distribution:")
    for lower, upper in brand_sim_ranges:
        count = len(potential_matches_df[
            (potential_matches_df['brand_similarity'] >= lower) & 
            (potential_matches_df['brand_similarity'] < upper)
        ])
        print(f"{lower:.2f}-{upper:.2f}: {count} matches")
    
    # Print some example matches for each range
    print("\nExample matches for each name similarity range:")
    for lower, upper in name_sim_ranges:
        matches = potential_matches_df[
            (potential_matches_df['name_similarity'] >= lower) & 
            (potential_matches_df['name_similarity'] < upper)
        ].head(3)
        if not matches.empty:
            print(f"\n{lower:.2f}-{upper:.2f} range examples:")
            print(matches[['base_name', 'sephora_name', 'name_similarity']].to_string())

def main():
    parser = argparse.ArgumentParser(description='Process and merge skincare product data.')
    parser.add_argument('--name-threshold', type=float, default=0.8,
                      help='Name similarity threshold for matching (default: 0.8)')
    parser.add_argument('--brand-threshold', type=float, default=0.85,
                      help='Brand similarity threshold for matching (default: 0.85)')
    parser.add_argument('--analyze-only', action='store_true',
                      help='Only analyze matches without performing merge')
    parser.add_argument('--show-all', action='store_true',
                      help='Show all matches instead of just top 10')
    
    args = parser.parse_args()
    
    # Load the data
    print("Loading data...")
    df_base = pd.read_csv("/Users/alexvalone/Desktop/Data_Science/DS_Q3/Visualization/skincare_project/data/cosmetics_skin_type/cosmetics.csv")
    df_sephora = pd.read_csv("/Users/alexvalone/Desktop/Data_Science/DS_Q3/Visualization/skincare_project/data/seph_skin_products/skincare_df.csv")
    
    # Standardize names in base DataFrame
    print("\nStandardizing names in base DataFrame...")
    df_base['product_name_std'] = df_base['Name'].apply(standardize_name)
    df_base['brand_name_std'] = df_base['Brand'].apply(standardize_brand_name).apply(standardize_name)
    df_base['product_type_std'] = df_base['Label'].apply(standardize_name)
    df_base['match_key'] = df_base['brand_name_std'] + "_" + df_base['product_name_std']
    
    # Standardize names in Sephora DataFrame
    print("Standardizing names in Sephora DataFrame...")
    df_sephora['product_name_std'] = df_sephora['name'].apply(standardize_name)
    df_sephora['brand_name_std'] = df_sephora['brand'].apply(standardize_brand_name).apply(standardize_name)
    df_sephora['match_key'] = df_sephora['brand_name_std'] + "_" + df_sephora['product_name_std']
    df_sephora.dropna(subset=['match_key'], inplace=True)
    
    # Process category files and add categories to Sephora DataFrame
    print("\nProcessing category files...")
    product_to_category_map = process_category_files()
    df_sephora['category'] = df_sephora['match_key'].map(product_to_category_map)
    
    # Find potential matches
    potential_matches, unmatched_base, unmatched_sephora = find_potential_matches(
        df_base, df_sephora, args.name_threshold, args.brand_threshold
    )
    
    if potential_matches:
        potential_matches_df = pd.DataFrame(potential_matches)
        potential_matches_df = potential_matches_df.sort_values(
            by=['match_type', 'name_similarity', 'brand_similarity'],
            ascending=[False, False, False]
        )
        
        # Print summary statistics
        exact_matches = potential_matches_df[potential_matches_df['match_type'] == 'exact']
        high_similarity = potential_matches_df[potential_matches_df['match_type'] == 'high_similarity']
        
        print(f"\nFound {len(potential_matches_df)} potential matches!")
        print(f"- {len(exact_matches)} exact matches (similarity = 1.0)")
        print(f"- {len(high_similarity)} high similarity matches (>{args.name_threshold})")
        
        # Analyze similarity distribution
        analyze_similarity_distribution(potential_matches_df)
        
        # Print matches
        print("\nExact matches:")
        pd.set_option('display.max_colwidth', None)
        if args.show_all:
            print(exact_matches[['base_name', 'base_brand', 'sephora_name', 'sephora_brand', 'name_similarity']].to_string())
        else:
            print(exact_matches[['base_name', 'base_brand', 'sephora_name', 'sephora_brand', 'name_similarity']].head(10).to_string())
        
        print("\nHigh similarity matches:")
        if args.show_all:
            print(high_similarity[['base_name', 'base_brand', 'sephora_name', 'sephora_brand', 'name_similarity']].to_string())
        else:
            print(high_similarity[['base_name', 'base_brand', 'sephora_name', 'sephora_brand', 'name_similarity']].head(10).to_string())
        
        # Save potential matches to CSV
        potential_matches_df.to_csv('/Users/alexvalone/Desktop/Data_Science/DS_Q3/Visualization/skincare_project/data/potential_product_matches.csv', index=False)
        print("\nSaved all potential matches to 'data/potential_product_matches.csv'")
        
        if not args.analyze_only:
            # Create match key mapping for all high similarity matches
            print("\nUpdating match keys for high similarity matches...")
            match_key_mapping = {}
            for _, row in potential_matches_df[potential_matches_df['should_merge']].iterrows():
                base_key = row['base_brand_std'] + "_" + row['base_name_std']
                sephora_key = row['sephora_brand_std'] + "_" + row['sephora_name_std']
                match_key_mapping[sephora_key] = base_key
            
            print(f"\nCreated mapping for {len(match_key_mapping)} matches")
            
            # Load the final merged DataFrame
            df_final_merged = pd.read_csv('/Users/alexvalone/Desktop/Data_Science/DS_Q3/Visualization/skincare_project/data/final_merged_products.csv')
            
            # Update match keys in Sephora DataFrame
            df_sephora['original_match_key'] = df_sephora['match_key']
            df_sephora['match_key'] = df_sephora['match_key'].map(lambda x: match_key_mapping.get(x, x))
            
            # Define columns to merge from Sephora
            columns_to_merge_from_sephora = [
                'match_key', 'clean_product',
                'n_of_reviews', 'n_of_loves', 'review_score',
                'size', 'category', 'price_per_ounce', 'return_on_reviews',
                'reviews_to_loves_ratio'
            ]
            existing_cols_in_sephora = [col for col in columns_to_merge_from_sephora if col in df_sephora.columns]
            
            # Merge with Sephora data using the updated match_keys
            print("\nMerging with Sephora data using updated match_keys...")
            df_sephora_subset_to_merge = df_sephora[existing_cols_in_sephora].drop_duplicates(subset=['match_key'], keep='first')
            
            df_final_merged = pd.merge(
                df_final_merged,
                df_sephora_subset_to_merge,
                on='match_key',
                how='left',
                suffixes=('', '_sephora')
            )
            
            # Print statistics about the merge
            print("\nMerge Statistics:")
            print(f"Original number of rows: {len(df_final_merged)}")
            print(f"Number of newly matched products: {df_final_merged['clean_product'].notna().sum()}")
            
            # Check unmatched products after merge
            print("\nAfter merge:")
            unmatched_base_after = df_base[~df_base['match_key'].isin(df_sephora['match_key'])]
            unmatched_sephora_after = df_sephora[~df_sephora['match_key'].isin(df_base['match_key'])]
            print(f"Unmatched products in base DataFrame: {len(unmatched_base_after)}")
            print(f"Unmatched products in Sephora DataFrame: {len(unmatched_sephora_after)}")
            print(f"Reduction in unmatched products: {len(unmatched_base) - len(unmatched_base_after)}")
            
            # Save the merged DataFrame
            print("\nSaving merged data...")
            df_final_merged.to_csv('/Users/alexvalone/Desktop/Data_Science/DS_Q3/Visualization/skincare_project/data/final_merged_products.csv', index=False)
            print("Saved merged data to 'data/final_merged_products.csv'")
    else:
        print("\nNo potential matches found with similarity threshold > 0.8")

if __name__ == "__main__":
    main()

    