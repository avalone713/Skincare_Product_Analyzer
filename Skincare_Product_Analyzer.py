import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash.exceptions import PreventUpdate
import numpy as np
import ast 

# --- Data Loading ---
try:
    df_final = pd.read_csv('data/processed/final_products_ingredients.csv', low_memory=False)
except FileNotFoundError:
    print("ERROR: 'data/processed/final_products_ingredients.csv' not found. Please ensure the file exists in the 'data' directory.")
    df_final = pd.DataFrame({
        'category': pd.Series(dtype='str'),
        'Brand': pd.Series(dtype='str'),
        'Name': pd.Series(dtype='str'),
        'Price': pd.Series(dtype='float'),
        'Ingredients': pd.Series(dtype='str'),
        'review_score': pd.Series(dtype='float'),
        'n_of_loves': pd.Series(dtype='int'),
        'n_of_reviews': pd.Series(dtype='int'),
        'paula_ingredient_details': pd.Series(dtype='str'),
        'Combination': pd.Series(dtype='int'), 
        'Dry': pd.Series(dtype='int'),
        'Normal': pd.Series(dtype='int'), 
        'Oily': pd.Series(dtype='int'),
        'Sensitive': pd.Series(dtype='int'),
        'clean_product': pd.Series(dtype='int')
    })

# Ensure essential columns exist
essential_cols = ['category', 'Brand', 'Name', 'Price', 'Ingredients', 'review_score', 'n_of_loves', 'n_of_reviews', 'paula_ingredient_details']
for col in essential_cols:
    if col not in df_final.columns:
        df_final[col] = None if col not in ['Price', 'review_score', 'n_of_loves', 'n_of_reviews'] else 0

skin_type_cols = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']

if 'Price' in df_final.columns:
    df_final['Price'] = pd.to_numeric(df_final['Price'], errors='coerce').fillna(0)
if 'review_score' in df_final.columns:
    df_final['review_score'] = pd.to_numeric(df_final['review_score'], errors='coerce').fillna(0)
if 'n_of_loves' in df_final.columns:
    df_final['n_of_loves'] = pd.to_numeric(df_final['n_of_loves'], errors='coerce').fillna(0)
if 'n_of_reviews' in df_final.columns:
    df_final['n_of_reviews'] = pd.to_numeric(df_final['n_of_reviews'], errors='coerce').fillna(0)

for col in ['review_score', 'n_of_loves', 'n_of_reviews', 'Price']:
    if col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')
        df_final.loc[df_final[col] == 0, col] = np.nan

# Use 'category' as the primary category column
categories = sorted(df_final['category'].dropna().unique()) if 'category' in df_final.columns else []
brands = sorted(df_final['Brand'].dropna().unique()) if 'Brand' in df_final.columns else []

min_price = int(df_final['Price'].min()) if not df_final.empty and df_final['Price'].notna().any() else 0
max_price = int(df_final['Price'].max()) if not df_final.empty and df_final['Price'].notna().any() else 100
max_price = max(min_price + 20, max_price) if max_price > min_price else min_price + 20
price_marks = {i: f'${i}' for i in range(min_price, max_price + 1, 20)}

allergen_options = [
    {'label': 'Added Fragrance (Parfum/Fragrance)', 'value': 'fragrance_parfum'},
    {'label': 'Common Fragrance Allergens', 'value': 'fragrance_components'},
    {'label': 'Parabens', 'value': 'parabens_group'},
    {'label': 'Sulfates (SLS/SLES)', 'value': 'sulfates_group'},
    {'label': 'Drying Alcohols', 'value': 'drying_alcohols'},
    {'label': 'Silicones', 'value': 'silicones_group'},
    {'label': 'Chemical Sunscreens', 'value': 'chemical_sunscreens_group'},
    {'label': 'Formaldehyde Releasers', 'value': 'formaldehyde_releasers_group'},
    {'label': 'MI/MCI (Methylisothiazolinone/Methylchloroisothiazolinone)', 'value': 'mi_mci_group'},
    {'label': 'Propylene Glycol', 'value': 'propylene_glycol_group'},
    {'label': 'Cocamidopropyl Betaine', 'value': 'cocamidopropyl_betaine_group'},
    {'label': 'Phenoxyethanol', 'value': 'phenoxyethanol_group'},
    {'label': 'Lanolin', 'value': 'lanolin_group'},
    {'label': 'Artificial Colorants (Synthetic Dyes)', 'value': 'artificial_colorants_group'},
    {'label': 'Mineral Oil & Petrolatum', 'value': 'mineral_oil_petrolatum_group'},
    {'label': 'Talc', 'value': 'talc_group'},
    {'label': 'BHA/BHT (Preservatives)', 'value': 'bha_bht_group'}
]
ALLERGEN_GROUPS = {
    'fragrance_parfum': ['fragrance', 'parfum'],
    'fragrance_components': ['linalool', 'limonene', 'citronellol', 'geraniol', 'citral', 'eugenol', 'coumarin', 'farnesol', 'hexyl cinnamal', 'hydroxycitronellal', 'isoeugenol', 'benzyl alcohol', 'benzyl benzoate', 'benzyl salicylate', 'anisyl alcohol', 'amyl cinnamal', 'cinnamyl alcohol', 'cinnamal', 'alpha-isomethyl ionone', 'methyl 2-octynoate', 'evernia prunastri', 'evernia furfuracea'],
    'parabens_group': ['paraben', 'methylparaben', 'ethylparaben', 'propylparaben', 'butylparaben', 'isobutylparaben', 'isopropylparaben'],
    'sulfates_group': ['sodium lauryl sulfate', 'sodium laureth sulfate', 'sls', 'sles', 'ammonium lauryl sulfate', 'ammonium laureth sulfate', 'als', 'ales', 'sodium C14-16 olefin sulfonate'],
    'drying_alcohols': ['alcohol denat.', 'sd alcohol', 'ethanol', 'isopropyl alcohol', 'alcohol'],
    'silicones_group': ['dimethicone', 'cyclomethicone', 'cyclopentasiloxane', 'cyclohexasiloxane','dimethiconol', 'phenyl trimethicone', 'amodimethicone', 'cyclotetrasiloxane','cetyl dimethicone', 'dimethicone copolyol', 'stearyl dimethicone', '-siloxane', '-cone'],
    'chemical_sunscreens_group': ['oxybenzone', 'avobenzone', 'octinoxate', 'ethylhexyl methoxycinnamate', 'octisalate', 'ethylhexyl salicylate', 'homosalate', 'octocrylene', 'benzophenone-3', 'benzophenone-4', 'ensulizole', 'phenylbenzimidazole sulfonic acid', 'ecamsule', 'terephthalylidene dicamphor sulfonic acid', 'drometrizole trisiloxane'],
    'formaldehyde_releasers_group': ['dmdm hydantoin', 'imidazolidinyl urea', 'diazolidinyl urea', 'quaternium-15', 'bronopol', '2-bromo-2-nitropropane-1,3-diol', '5-bromo-5-nitro-1,3-dioxane','sodium hydroxymethylglycinate', 'methenamine', 'benzylhemiformal'],
    'mi_mci_group': ['methylisothiazolinone', 'mi', 'mit', 'methylchloroisothiazolinone', 'mci', 'mcit', 'cmIT'],
    'propylene_glycol_group': ['propylene glycol', 'pg', '1,2-propanediol'],
    'cocamidopropyl_betaine_group': ['cocamidopropyl betaine', 'capb'],
    'phenoxyethanol_group': ['phenoxyethanol'],
    'lanolin_group': ['lanolin', 'lanolin alcohol', 'adeps lanae', 'lanolin cera', 'lanolin oil', 'hydrogenated lanolin', 'wool fat', 'wool wax'],
    'artificial_colorants_group': ['ci 19140', 'ci 42090', 'ci 16035', 'ci 17200', 'ci 60730', 'ci 15850', 'ci 45410','fd&c yellow no. 5', 'fd&c blue no. 1', 'fd&c red no. 40','d&c red no. 33', 'ext. d&c violet no. 2', 'd&c red no. 6', 'd&c red no. 27','yellow 5', 'blue 1', 'red 40', 'red 33', 'violet 2', 'red 6', 'red 27'],
    'mineral_oil_petrolatum_group': ['mineral oil', 'paraffinum liquidum', 'liquid paraffin', 'huile minerale','petrolatum', 'white petrolatum', 'petroleum jelly', 'vaseline'],
    'talc_group': ['talc', 'talcum powder', 'cosmetic talc'],
    'bha_bht_group': ['bha', 'butylated hydroxyanisole', 'bht', 'butylated hydroxytoluene']
}
INTERACTION_RULES = [
    {'ingredients': ['retinol', 'glycolic acid'], 'warning': 'Interaction: Retinol + Glycolic Acid (AHA).'},
    {'ingredients': ['retinol', 'salicylic acid'], 'warning': 'Interaction: Retinol + Salicylic Acid (BHA).'},
    {'ingredients': ['ascorbic acid', 'niacinamide'], 'warning': 'Interaction: Vit C (Ascorbic) + Niacinamide.'},
    {'ingredients': ['benzoyl peroxide', 'retinol'], 'warning': 'Interaction: Benzoyl Peroxide + Retinol (and other retinoids like tretinoin, adapalene). Can deactivate each other (especially tretinoin) and increase irritation. Some forms of adapalene are stable with BPO. Generally best to alternate (e.g., BPO in AM, Retinol in PM) or use specialized combination products.'},
    {'ingredients': ['benzoyl peroxide', 'tretinoin'], 'warning': 'Interaction: Benzoyl Peroxide + Tretinoin. High risk of deactivation of tretinoin and increased irritation. Avoid simultaneous use unless specifically formulated together.'},
    {'ingredients': ['benzoyl peroxide', 'adapalene'], 'warning': 'Interaction: Benzoyl Peroxide + Adapalene. Generally more stable together than BPO + other retinoids, but still potential for irritation. Often formulated together in products like Epiduo.'},
    {'ingredients': ['ascorbic acid', 'glycolic acid'], 'warning': 'Interaction: Vit C (L-Ascorbic Acid forms) + Glycolic Acid (AHA). Potential for increased irritation, photosensitivity, and compromised skin barrier, especially at high concentrations or low pH. Use with caution, ensure stable formulations, or alternate.'},
    {'ingredients': ['ascorbic acid', 'lactic acid'], 'warning': 'Interaction: Vit C (L-Ascorbic Acid forms) + Lactic Acid (AHA). Potential for increased irritation, photosensitivity, and compromised skin barrier. Use with caution or alternate.'},
    {'ingredients': ['ascorbic acid', 'salicylic acid'], 'warning': 'Interaction: Vit C (L-Ascorbic Acid forms) + Salicylic Acid (BHA). Potential for increased irritation and dryness. Use with caution or alternate.'},
    {'ingredients': ['copper peptides', 'ascorbic acid'], 'warning': 'Interaction: Copper Peptides + Vit C (Direct forms like L-Ascorbic Acid). May oxidize and reduce efficacy of both ingredients. Best to use at different times of day or use Vitamin C derivatives.'},
    {'ingredients': ['benzoyl peroxide', 'ascorbic acid'], 'warning': 'Interaction: Benzoyl Peroxide + Vit C (L-Ascorbic Acid). Benzoyl peroxide can oxidize L-Ascorbic Acid, reducing its effectiveness. Apply at different times of day.'},
    {'ingredients': ['retinol', 'ascorbic acid'], 'warning': 'Interaction: Retinol + Vit C (L-Ascorbic Acid). Can increase irritation due to different pH requirements for optimal stability/penetration and combined exfoliant effects. Often recommended to use at different times of day (e.g., Vit C in AM, Retinol in PM).'},
    {'ingredients': ['alpha hydroxy acid', 'beta hydroxy acid'], 'warning': 'Interaction: AHA (e.g., Glycolic, Lactic) + BHA (Salicylic Acid). Using multiple strong exfoliants together can lead to over-exfoliation, irritation, and damaged skin barrier. Introduce slowly and monitor skin response; often better to alternate.'}
]

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# --- App Layout ---
app.layout = html.Div([
    # Add Store components for data persistence
    dcc.Store(id='store-selected-for-comparison-names'),
    
    # Application Header and Instructions
    html.Div([
        html.H2("Skincare Product Analysis Tool", style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#2c3e50'}),
        html.Div([
            html.P([
                "Welcome to the Skincare Product Analysis Tool! This application helps you analyze and compare skincare products based on their ingredients, reviews, and price points. ",
                "Use the filters on the left to find products that match your preferences, then explore detailed ingredient analysis and comparisons."
            ], style={'textAlign': 'center', 'marginBottom': '20px', 'color': '#34495e'})
        ], style={'maxWidth': '1000px', 'margin': '0 auto'})
    ], style={'marginBottom': '20px'}),
    
    html.Div([
        html.Div([
            html.H3("Search & Select Products", className="content-card-title"),
            html.Label("Ingredients to Exclude:", style={'fontWeight': 400, 'fontSize': '1rem', 'color': '#222', 'marginBottom': '4px'}),
            dcc.Input(id='base-exclude-ingredients', type='text', style={'width': '100%', 'marginBottom': '10px', 'fontSize': '0.95rem', 'boxSizing': 'border-box', 'padding': '6px'}),
            html.Label("Category:", style={'fontWeight': 400, 'fontSize': '0.95rem', 'marginBottom': '4px'}),
            dcc.Dropdown(
                id='base-category-dropdown',
                options=[{'label': cat, 'value': cat} for cat in categories],
                placeholder="Select main category...",
                multi=True,
                style={'width': '100%', 'marginBottom': '10px', 'fontSize': '0.95rem'}
            ),
            html.Label("Brand:", style={'fontWeight': 400, 'fontSize': '0.95rem', 'marginBottom': '4px'}),
            dcc.Dropdown(
                id='base-brand-dropdown',
                options=[{'label': b, 'value': b} for b in brands],
                placeholder="Select brand...",
                multi=True,
                style={'width': '100%', 'marginBottom': '10px', 'fontSize': '0.95rem'}
            ),
            html.Label("Skin Type:", style={'fontWeight': 400, 'fontSize': '0.95rem', 'marginBottom': '4px'}),
            dcc.Checklist(id='skin-type-checklist', options=[{'label': st, 'value': st} for st in skin_type_cols], value=[], inline=True, style={'marginBottom': '10px', 'fontSize': '0.95rem'}),
            html.Label("Clean Product:", style={'fontWeight': 400, 'fontSize': '0.95rem', 'marginBottom': '4px'}),
            dcc.Checklist(id='clean-product-checklist', options=[{'label': 'Yes', 'value': 1}], value=[], style={'marginBottom': '10px', 'fontSize': '0.95rem'}),
            html.Button('Apply Filters', id='btn-initial-search', n_clicks=0, style={'marginBottom': '18px', 'padding': '6px 12px', 'fontSize': '0.95rem'}),
            html.Label("Select Product(s):", style={'fontWeight': 400, 'fontSize': '0.95rem', 'marginBottom': '4px'}),
            dcc.Dropdown(id='product-search-dropdown-single', options=[],multi=True, placeholder="Type to search for a product...", searchable=True, style={'marginBottom': '10px', 'fontSize': '0.95rem'})
        ], className="filter-column", style={'width': '300px'}),
        html.Div([
            dcc.Tabs(
                id='main-tabs',
                value='reviews-price-tab',
                children=[
                    dcc.Tab(label='Compare Price & Reviews', value='reviews-price-tab', className="custom-tab"),
                    dcc.Tab(label='In-Depth Ingredient Analysis', value='ingredient-analysis-tab', className="custom-tab")
                ], 
                className="custom-tabs-container", 
                style={'marginBottom': '0'}
            ),
            html.Div(id='main-tab-content', className="content-card figure-area", style={'padding': '20px', 'width': '100%', 'height': 'fit-content'})
        ], className="main-content-card", style={'flex': '1', 'height': 'fit-content'})
    ], style={'display': 'flex','columnGap': '32px'})
])
# --- Helper Functions ---
def split_ingredients(ingredients_str):
    """
    Split a semicolon-separated ingredients string into a list of individual ingredients.
    
    Args:
        ingredients_str (str): String containing ingredients separated by semicolons
        
    Returns:
        list: List of individual ingredients, stripped and converted to lowercase
    """
    if pd.isna(ingredients_str) or not str(ingredients_str).strip(): return []
    return [ing.strip().lower() for ing in str(ingredients_str).split(';') if ing.strip()]

def parse_paula_details(details_val):
    """
    Parse Paula's Choice ingredient details from string or list format into a list of dictionaries.
    
    Args:
        details_val (str/list): String representation of list or actual list of ingredient details
        
    Returns:
        list: List of dictionaries containing ingredient information
    """
    if isinstance(details_val, list):
        if all(isinstance(item, dict) for item in details_val):
            return details_val
        else:
            return []
    if isinstance(details_val, str):
        try:
            parsed = ast.literal_eval(details_val)
            if isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                return parsed
        except Exception:
            pass
    return []

def get_product_warnings(product_ingredients_set, selected_allergens_groups_keys):
    """
    Generate warnings for a product based on its ingredients and selected allergen groups.
    
    Args:
        product_ingredients_set (set): Set of ingredients in the product
        selected_allergens_groups_keys (list): List of allergen group keys to check
        
    Returns:
        list: List of warning messages for allergens and ingredient interactions
    """
    warnings = []
    
    # Check for allergens in selected groups
    for allergen_key in selected_allergens_groups_keys: 
        if allergen_key in ALLERGEN_GROUPS:
            keywords = ALLERGEN_GROUPS[allergen_key]
            found = False

            # Check ingredients against allergen keywords
            if allergen_key == 'parabens_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'silicones_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'drying_alcohols':
                if any(alc_keyword in product_ingredients_set for alc_keyword in keywords):
                    found = True
            elif allergen_key == 'chemical_sunscreens_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'formaldehyde_releasers_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'mi_mci_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'propylene_glycol_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True    
            elif allergen_key == 'cocamidopropyl_betaine_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'phenoxyethanol_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'lanolin_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'artificial_colorants_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'mineral_oil_petrolatum_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'talc_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            elif allergen_key == 'bha_bht_group':
                if any(any(k in ing for k in keywords) for ing in product_ingredients_set):
                    found = True
            else:
                for ing in product_ingredients_set:
                    if any(keyword in ing for keyword in keywords):
                        found = True
                        break

            if found:
                label = allergen_key.replace('_', ' ').title()
                for opt in allergen_options:
                    if opt['value'] == allergen_key:
                        label = opt['label']
                        break
                warnings.append(f"Contains: {label.split(' (')[0]}")

    # Check for ingredient interactions
    for rule in INTERACTION_RULES:
        rule_ingredients_found = []
        for rule_ing in rule['ingredients']:
            if any(rule_ing.lower() in ing.lower() for ing in product_ingredients_set):
                rule_ingredients_found.append(rule_ing)
        
        if len(rule_ingredients_found) == len(rule['ingredients']):
            warnings.append(html.Div([
                html.Div([
                    html.I(className="fas fa-exclamation-circle", style={'marginRight': '8px', 'color': '#721c24'}),
                    html.Strong("Interaction Warning", style={'color': '#721c24'})
                ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
                html.Div(rule['warning'], style={'fontSize': '0.9em', 'color': '#666'})
            ], style={
                'backgroundColor': '#f8d7da',
                'border': '1px solid #f5c6cb',
                'borderRadius': '4px',
                'padding': '12px',
                'marginBottom': '8px'
            }))
            
    return warnings

def get_filtered_df(exclude_ings_str, category_vals, brand_vals, skin_types, clean_product_flag):
    """
    Filter the product dataframe based on user-selected criteria.
    
    Args:
        exclude_ings_str (str): Comma-separated string of ingredients to exclude
        category_vals (list): List of selected categories
        brand_vals (list): List of selected brands
        skin_types (list): List of selected skin types
        clean_product_flag (bool): Whether to show only clean products
        
    Returns:
        DataFrame: Filtered dataframe matching the criteria
    """
    filtered_df = df_final.copy()
    
    # Filter by excluded ingredients
    if exclude_ings_str:
        exclude_list = [ing.strip().lower() for ing in exclude_ings_str.split(',') if ing.strip()]
        if exclude_list:
            def check_excluded(ingredients_text):
                if pd.isna(ingredients_text): return True
                product_ings = split_ingredients(ingredients_text)
                return not any(excluded_ing in product_ings for excluded_ing in exclude_list)
            filtered_df = filtered_df[filtered_df['Ingredients'].apply(check_excluded)]
    
    # Filter by category
    if category_vals:
        filtered_df = filtered_df[filtered_df['category'].isin(category_vals)]
    
    # Filter by brand
    if brand_vals:
        filtered_df = filtered_df[filtered_df['Brand'].isin(brand_vals)]
    
    # Filter by skin type
    if skin_types:
        filtered_df = filtered_df[filtered_df[skin_types].any(axis=1)]
    
    # Filter by clean product status
    if clean_product_flag:
        if 'Clean_Product_Boolean' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Clean_Product_Boolean'] == 1]
            
    return filtered_df

# --- Callback Functions ---
@app.callback(
    [Output('product-search-dropdown-single', 'options'),
     Output('product-search-dropdown-single', 'value')],
    Input('btn-initial-search', 'n_clicks'),
    [State('base-exclude-ingredients', 'value'),
     State('base-category-dropdown', 'value'),
     State('base-brand-dropdown', 'value'),
     State('skin-type-checklist', 'value'),
     State('clean-product-checklist', 'value')],
    prevent_initial_call=True
)
def update_product_dropdown_options(n_clicks, exclude_ings_str, category_vals, brand_vals, skin_types, clean_product_flag_list):
    """
    Update the product dropdown options based on filter selections.
    
    Args:
        n_clicks (int): Number of times the search button has been clicked
        exclude_ings_str (str): Ingredients to exclude
        category_vals (list): Selected categories
        brand_vals (list): Selected brands
        skin_types (list): Selected skin types
        clean_product_flag_list (list): Clean product filter selection
        
    Returns:
        tuple: (dropdown options, selected value)
    """
    if n_clicks == 0 or df_final.empty:
        raise PreventUpdate
        
    clean_flag_bool = True if clean_product_flag_list and 1 in clean_product_flag_list else False
    filtered_df = get_filtered_df(exclude_ings_str, category_vals, brand_vals, skin_types, clean_flag_bool)
    
    if 'review_score' in filtered_df.columns and 'n_of_loves' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['review_score'].notna() & filtered_df['n_of_loves'].notna()]
    
    options = [{'label': f"{row['Name']} ({row['Brand']})", 'value': row['Name']} for _, row in filtered_df.iterrows()]
    return options, []

# --- Callbacks ---

## ---- Initial Product Selection and Filtering ---- ##
#Initial Product Selection Filters
@app.callback(
    [Output('store-initial-filtered-product-indices', 'data'),
     Output('initial-search-results-info', 'children')],
    Input('btn-initial-search', 'n_clicks'),
    [State('base-include-ingredients', 'value'),
     State('base-exclude-ingredients', 'value'),
     State('base-category-dropdown', 'value'),
     State('base-brand-dropdown', 'value')],
    prevent_initial_call=True
)
def apply_initial_filters(n_clicks, include_ings_str, exclude_ings_str, category_val, brand_val):
    """
    Apply initial filters to the product dataset and return filtered indices.
    
    Args:
        n_clicks (int): Number of times the search button has been clicked
        include_ings_str (str): Ingredients to include
        exclude_ings_str (str): Ingredients to exclude
        category_val (str): Selected category
        brand_val (str): Selected brand
        
    Returns:
        tuple: (filtered indices, info message)
    """
    if n_clicks == 0 or df_final.empty:
        raise PreventUpdate

    filtered_df = df_final.copy()

    # Apply ingredient filters
    if exclude_ings_str:
        exclude_list = [ing.strip().lower() for ing in exclude_ings_str.split(',') if ing.strip()]
        if exclude_list:
            def check_excluded(ingredients_text):
                if pd.isna(ingredients_text): return True
                product_ings = split_ingredients(ingredients_text)
                return not any(excluded_ing in product_ings for excluded_ing in exclude_list)
            filtered_df = filtered_df[filtered_df['Ingredients'].apply(check_excluded)]

    if include_ings_str:
        include_list = [ing.strip().lower() for ing in include_ings_str.split(',') if ing.strip()]
        if include_list:
            def check_included(ingredients_text):
                if pd.isna(ingredients_text): return False
                product_ings = split_ingredients(ingredients_text)
                return all(included_ing in product_ings for included_ing in include_list)
            filtered_df = filtered_df[filtered_df['Ingredients'].apply(check_included)]

    # Apply category and brand filters
    if category_val:
        filtered_df = filtered_df[filtered_df['category'] == category_val]
    if brand_val:
        filtered_df = filtered_df[filtered_df['Brand'] == brand_val]

    num_results = len(filtered_df)
    info_text = f"Found {num_results} products matching your initial criteria. Proceed to 'Product Analysis & Visuals' tab."
    return filtered_df.index.tolist(), info_text


# Apply Advanced Filters
@app.callback(
    [Output('store-advanced-filtered-product-indices', 'data'),
     Output('advanced-filter-results-info', 'children')],
    Input('btn-advanced-filter', 'n_clicks'),
    [State('store-initial-filtered-product-indices', 'data'),
     State('skin-type-checklist', 'value'),
     State('clean-product-checklist', 'value'),
     State('adv-price-slider', 'value')],
    prevent_initial_call=True
)
def apply_advanced_filters(n_clicks, initial_indices, skin_types, clean_product_flag, price_range_adv):
    """
    Apply advanced filters to the initially filtered dataset.
    
    Args:
        n_clicks (int): Number of times the advanced filter button has been clicked
        initial_indices (list): Indices from initial filtering
        skin_types (list): Selected skin types
        clean_product_flag (list): Clean product filter selection
        price_range_adv (list): Price range [min, max]
        
    Returns:
        tuple: (filtered indices, info message)
    """
    if n_clicks == 0 or initial_indices is None or df_final.empty:
        raise PreventUpdate

    current_df = df_final.loc[initial_indices].copy()

    # Apply skin type filter
    if skin_types:
        current_df = current_df[current_df[skin_types].any(axis=1)]

    # Apply price filter
    if price_range_adv:
        current_df = current_df[(current_df['Price'] >= price_range_adv[0]) & (current_df['Price'] <= price_range_adv[1])]

    num_results = len(current_df)
    info_text = f"Applied advanced filters. {num_results} products available for selection."
    return current_df.index.tolist(), info_text


# Store selected products for comparison
@app.callback(
    Output('store-selected-for-comparison-names', 'data'),
    Input('product-search-dropdown-single', 'value')
)
def store_products_for_comparison(selected_products):
    """
    Store selected products for comparison.
    
    Args:
        selected_products (str/list): Selected product name(s)
        
    Returns:
        list: List of selected product names
    """
    if not selected_products:
        return []
    if isinstance(selected_products, str):
        return [selected_products]
    return selected_products

## ---- Reviews and Price Comparison Tab ---- ##

# Generate Comparison Visualizations 
@app.callback(
    Output('price-review-plot', 'figure'),
    [Input('price-review-plot-type', 'value'),
     Input('price-distribution-group', 'value'),
     Input('product-search-dropdown-single', 'value')],
    [State('base-exclude-ingredients', 'value'),
     State('base-category-dropdown', 'value'),
     State('base-brand-dropdown', 'value'),
     State('skin-type-checklist', 'value'),
     State('clean-product-checklist', 'value')],
    prevent_initial_call=True
)
def update_price_review_plot(plot_type, group_by, selected_products, exclude_ings_str, category_vals, brand_vals, skin_types, clean_product_flag_list):
    """
    Generate price and review comparison plots.
    
    Args:
        plot_type (str): Type of plot ('scatter' or 'box')
        group_by (str): Grouping variable for box plot
        selected_products (list): Selected products to display
        exclude_ings_str (str): Ingredients to exclude
        category_vals (list): Selected categories
        brand_vals (list): Selected brands
        skin_types (list): Selected skin types
        clean_product_flag_list (list): Clean product filter selection
        
    Returns:
        Figure: Plotly figure object
    """
    clean_flag_bool = True if clean_product_flag_list and 1 in clean_product_flag_list else False
    filtered_df = get_filtered_df(exclude_ings_str, category_vals, brand_vals, skin_types, clean_flag_bool)
    
    if filtered_df.empty:
        return go.Figure().update_layout(
            title_x=0.5,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    
    if selected_products:
        if isinstance(selected_products, list):
            plot_df = filtered_df[filtered_df['Name'].isin(selected_products)]
        else:
            plot_df = filtered_df[filtered_df['Name'] == selected_products]
    else:
        plot_df = filtered_df
        
    if plot_df.empty:
        return go.Figure().update_layout(
            title_x=0.5,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    
    if plot_type == 'scatter':
        # Filter for required fields and create bins for n_of_loves
        plot_df = plot_df[plot_df['review_score'].notna() & plot_df['n_of_loves'].notna()].copy()
        required_fields = ['review_score', 'n_of_loves', 'n_of_reviews', 'Price']
        plot_df = plot_df.dropna(subset=required_fields).copy()
        
        if plot_df.empty:
            return go.Figure().update_layout(
                title_x=0.5,
                paper_bgcolor="rgba(0,0,0,0)"
            )
        
        # Define color sequence and bin order
        cerulean_seq_5 = [
            "#0f0f40",  # Very High
            "#24769a",  # High
            "#32c0cf",  # Medium
            "#7ad9df",  # Low
            "#e6fff7"  # Very Low
        ]

        bin_order = [
            'Very High (26k - 193k)',
            'High (13k - 26k)',
            'Medium (7k - 13k)',
            'Low (3k - 7k)',
            'Very Low (0 - 3k)'
        ]
        
        # Create scatter plot
        fig = px.scatter(
            plot_df,
            x='review_score',
            y='Price',
            size='n_of_reviews',
            color='n_of_loves_bin',
            hover_name='Name',
            hover_data=['Brand', 'Price', 'review_score', 'n_of_loves', 'n_of_reviews', 'n_of_loves_bin'],
            color_discrete_sequence=cerulean_seq_5,
            labels={'n_of_loves_bin': 'Popularity', 'review_score': 'Review Score', 'n_of_reviews': 'Number of Reviews'},
            category_orders={'n_of_loves_bin': bin_order}
        )
        
        fig.update_traces(marker=dict(line=dict(width=1, color='black')))
        fig.update_layout(
            title_text="",
            autosize=True,
            height=500,  # Make the plot smaller
            font_family="Inter, Arial, sans-serif",
            font_size=18,  # Increase base font size
            margin=dict(l=10, r=10, t=10, b=10),  # Keep small margins
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.02,
                font=dict(size=14),  # Smaller legend font
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            xaxis=dict(
                title=dict(
                    font=dict(size=18),  # Larger axis title font
                    standoff=10
                ),
                tickfont=dict(size=16),  # Larger tick font
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title=dict(
                    font=dict(size=18),  # Larger axis title font
                    standoff=10
                ),
                tickfont=dict(size=16),  # Larger tick font
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            )
        )
    else:
        # Create box plot
        fig = px.box(
            plot_df,
            x=group_by,
            y='Price',
            color=group_by,
            points='outliers',
            title=f'Price Distribution by {group_by.title()}',
            labels={group_by: group_by.title(), 'Price': 'Price ($)'}
        )
        
        # Update layout
        fig.update_layout(
            showlegend=False,
            xaxis_title=f'{group_by.title()}',
            yaxis_title='Price ($)',
            font_family="Inter, Arial, sans-serif",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title=dict(
                    font=dict(size=12),
                    standoff=10
                ),
                tickfont=dict(size=12),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            ),
            yaxis=dict(
                title=dict(
                    font=dict(size=12),
                    standoff=10
                ),
                tickfont=dict(size=12),
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)'
            )
        )
    
    return fig

# --- Main Tab Content Callback ---
@app.callback(
    Output('main-tab-content', 'children'),
    Input('main-tabs', 'value')
)
def render_main_tab_content(tab_value):
    """
    Render the content for the main tab based on the selected tab value.
    
    Args:
        tab_value (str): The selected tab value ('reviews-price-tab' or 'ingredient-analysis-tab')
        
    Returns:
        Div: Dash HTML component containing the tab content
    """
    if tab_value == 'reviews-price-tab':
        return html.Div([
            html.Div([
                html.H4("Product Analysis", className="figure-header", style={'marginBottom': '10px'}),
                html.Div([
                    html.P([
                        "Compare products based on their price points and review scores. ",
                        "The scatter plot shows the relationship between price and review scores, with bubble size indicating the number of reviews. ",
                        "Use the box plot to see price distributions across different categories or brands."
                    ], style={'marginBottom': '15px', 'color': '#666'}),
                    html.Div([
                        html.Div(
                            dcc.RadioItems(
                                id='price-review-plot-type',
                                options=[
                                    {'label': 'Price vs. Review Score', 'value': 'scatter'},
                                    {'label': 'Price Distribution', 'value': 'box'}
                                ],
                                value='scatter',
                                inline=True,
                                className="dash-radioitems",
                                style={'paddingLeft': '12px'} 
                            ), 
                            style={'float': 'left'}
                        ),
                        html.Div(
                            dcc.RadioItems(
                                id='price-distribution-group',
                                options=[
                                    {'label': 'By Category', 'value': 'category'},
                                    {'label': 'By Brand', 'value': 'Brand'}
                                ],
                                value='category',
                                inline=False,
                                className="dash-radioitems",
                                style={'fontSize': '0.9rem', 'paddingRight': '12px'}
                            ),
                            id='price-distribution-group-container',
                            style={'display': 'none', 'float': 'right', 'marginTop': '30px'}
                        )
                    ]),
                    html.Div(style={'clear': 'both'})
                ], style={'marginBottom': '20px', 'position': 'relative', 'minHeight': '60px'}),
                html.Div([
                    dcc.Loading(children=[
                        dcc.Graph(
                            id='price-review-plot',
                            style={
                                'width': '100%',
                                'height': '100%',
                                'margin': '0',
                                'padding': '0'
                            },
                            config={
                                'displayModeBar': True,
                                'displaylogo': False,
                                'responsive': True
                            }
                        )
                    ])
                ], style={'clear': 'both'})
            ])
        ])
    elif tab_value == 'ingredient-analysis-tab':
        return html.Div([ 
            html.Div([
                html.H4("Ingredient Analysis", className="figure-header", style={'marginBottom': '10px'}),
                html.P([
                    "Analyze the ingredients in your selected products. Choose from three types of analysis:"
                ], style={'marginBottom': '15px', 'color': '#666'}),
                html.Ul([
                    html.Li([
                        html.Strong("Allergens & Interactions: "),
                        "Check for potential allergens and ingredient interactions that might affect your skin."
                    ]),
                    html.Li([
                        html.Strong("Ingredient Details & Functions: "),
                        "View detailed information about each ingredient, including its functions and benefits."
                    ]),
                    html.Li([
                        html.Strong("Formulation Profile: "),
                        "See how ingredients are categorized and their relationships in the product formulation."
                    ])
                ], style={'marginBottom': '20px', 'paddingLeft': '20px', 'color': '#666'})
            ], style={'marginBottom': '20px'}),
            html.Label("Select Product for Ingredient Analysis:", style={'fontSize': '1rem', 'fontWeight': 500, 'marginBottom': '8px', 'paddingTop': '8px'}),
            dcc.Dropdown(
                id='ia-product-selector', 
                value=None,
                placeholder="Products from filter appear here...",
                style={'width': '100%', 'marginBottom': '20px'}
            ),
            html.Label("Choose Analysis Type:", style={'fontSize': '1rem', 'fontWeight': 500, 'marginBottom': '8px'}),
            dcc.RadioItems(
                id='ia-analysis-type-selector',
                options=[
                    {'label': 'Allergens & Interactions', 'value': 'allergens_interactions'},
                    {'label': 'Ingredient Details & Functions', 'value': 'details_functions'},
                    {'label': 'Formulation Profile', 'value': 'composition'}
                ],
                value='details_functions',
                inline=True, 
                className="dash-radioitems",
                style={'marginBottom': '20px'}
            ),
            html.Div([
                dcc.Dropdown(
                    id='ia-allergen-group-dropdown',
                    options=allergen_options,
                    multi=True,
                    value=[],
                    placeholder="Select allergen groups to filter...",
                    style={'width': '100%', 'marginBottom': '16px'}
                )
            ], id='allergen-dropdown-container', style={'display': 'none'}),
            dcc.Loading(html.Div(id='ia-analysis-output-area')) 
        ], style={'overflowY': 'auto','padding': '8px 8px 8px 8px'})
    return html.P("Select a tab.")

# --- Ingredient Analysis Sub-tab Content Callback ---
@app.callback(
    [Output('ia-product-selector', 'options'),
     Output('ia-product-selector', 'value')],
    Input('main-tabs', 'value'),
    State('ia-product-selector', 'value'),
    prevent_initial_call=False
)
def populate_ia_product_selector(tab_value, current_value):
    """
    Populate the ingredient analysis product selector dropdown.
    
    Args:
        tab_value (str): The selected tab value
        current_value (str): Currently selected product value
        
    Returns:
        tuple: (dropdown options, selected value)
    """
    options = []
    if tab_value != 'ingredient-analysis-tab':
        return options, None
    for _, row in df_final.iterrows():
        paula_details_list = parse_paula_details(row.get('paula_ingredient_details'))
        if paula_details_list and isinstance(paula_details_list, list) and len(paula_details_list) > 0:
            options.append({'label': f"{row['Name']} ({row['Brand']})", 'value': row['Name']})
    values = [opt['value'] for opt in options]
    if current_value in values:
        return options, current_value
    return options, (values[0] if values else None)

@app.callback(
    Output('ia-analysis-output-area', 'children'),
    [Input('ia-product-selector', 'value'),
     Input('ia-analysis-type-selector', 'value'),
     Input('ia-allergen-group-dropdown', 'value')],
    [State('base-exclude-ingredients', 'value')],
    prevent_initial_call=True
)
def update_ingredient_analysis_display(selected_product_name, analysis_type, selected_allergen_groups, exclude_ings_str):
    """
    Update the ingredient analysis display based on selected product and analysis type.
    
    Args:
        selected_product_name (str): Name of the selected product
        analysis_type (str): Type of analysis to perform
        selected_allergen_groups (list): Selected allergen groups to check
        exclude_ings_str (str): Ingredients to exclude
        
    Returns:
        Div: Dash HTML component containing the analysis results
    """
    if not selected_product_name:
        return html.Div([
            html.P("Please select a product for analysis.", style={'color': '#666', 'fontStyle': 'italic'})
        ])
    if not analysis_type:
        return html.Div([
            html.P("Please select an analysis type.", style={'color': '#666', 'fontStyle': 'italic'})
        ])

    product_data_row_df = df_final[df_final['Name'] == selected_product_name]
    if product_data_row_df.empty: 
        return html.Div([
            html.P(f"Details for product '{selected_product_name}' not found.", style={'color': '#dc3545'})
        ])

    product_data_row = product_data_row_df.iloc[0]

    if analysis_type == 'allergens_interactions':
        # Generate allergen and interaction warnings
        paula_details_list = parse_paula_details(product_data_row.get('paula_ingredient_details'))
        product_ingredients = set(split_ingredients(product_data_row.get('Ingredients', '')))
        warnings = []
        
        # Add context about allergen analysis
        context_div = html.Div([
            html.P([
                "This analysis checks for potential allergens and ingredient interactions in your selected product. ",
                "Select specific allergen groups to check, or view all potential allergens found in the product."
            ], style={'marginBottom': '15px', 'color': '#666'}),
            html.P([
                html.Strong("Note: "),
                "The presence of these ingredients doesn't necessarily mean they will cause issues. ",
                "Individual reactions vary, and many of these ingredients are well-tolerated by most people."
            ], style={'marginBottom': '20px', 'color': '#666', 'fontStyle': 'italic'})
        ])

        # Generate warnings based on selected allergen groups
        if selected_allergen_groups:
            for allergen_key in selected_allergen_groups:
                if allergen_key in ALLERGEN_GROUPS:
                    keywords = ALLERGEN_GROUPS[allergen_key]
                    if any(any(k in ing for k in keywords) for ing in product_ingredients):
                        label = next((opt['label'] for opt in allergen_options if opt['value'] == allergen_key), allergen_key)
                        warnings.append(html.Div([
                            html.Div([
                                html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px', 'color': '#856404'}),
                                html.Strong(f"Contains: {label.split(' (')[0]}", style={'color': '#856404'})
                            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
                            html.Div([
                                html.Span("Allergen keywords: ", style={'fontWeight': 500}),
                                html.Span(', '.join(keywords), style={'fontStyle': 'italic'})
                            ], style={'fontSize': '0.9em', 'color': '#666'})
                        ], style={
                            'backgroundColor': '#fff3cd',
                            'border': '1px solid #ffeeba',
                            'borderRadius': '4px',
                            'padding': '12px',
                            'marginBottom': '8px'
                        }))
        else:
            # Show warnings for all allergen groups present in the product
            for allergen_key, keywords in ALLERGEN_GROUPS.items():
                if any(any(k in ing for k in keywords) for ing in product_ingredients):
                    label = next((opt['label'] for opt in allergen_options if opt['value'] == allergen_key), allergen_key)
                    warnings.append(html.Div([
                        html.Div([
                            html.I(className="fas fa-exclamation-triangle", style={'marginRight': '8px', 'color': '#856404'}),
                            html.Strong(f"Contains: {label.split(' (')[0]}", style={'color': '#856404'})
                        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
                        html.Div([
                            html.Span("Allergen keywords: ", style={'fontWeight': 500}),
                            html.Span(', '.join(keywords), style={'fontStyle': 'italic'})
                        ], style={'fontSize': '0.9em', 'color': '#666'})
                    ], style={
                        'backgroundColor': '#fff3cd',
                        'border': '1px solid #ffeeba',
                        'borderRadius': '4px',
                        'padding': '12px',
                        'marginBottom': '8px'
                    }))

        # Generate interaction warnings
        interaction_warnings = []
        for rule in INTERACTION_RULES:
            rule_ingredients_found = []
            for rule_ing in rule['ingredients']:
                if any(rule_ing.lower() in ing.lower() for ing in product_ingredients):
                    rule_ingredients_found.append(rule_ing)
            
            if len(rule_ingredients_found) == len(rule['ingredients']):
                interaction_warnings.append(html.Div([
                    html.Div([
                        html.I(className="fas fa-exclamation-circle", style={'marginRight': '8px', 'color': '#721c24'}),
                        html.Strong("Interaction Warning", style={'color': '#721c24'})
                    ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
                    html.Div(rule['warning'], style={'fontSize': '0.9em', 'color': '#666'})
                ], style={
                    'backgroundColor': '#f8d7da',
                    'border': '1px solid #f5c6cb',
                    'borderRadius': '4px',
                    'padding': '12px',
                    'marginBottom': '8px'
                }))

        if not warnings and not interaction_warnings:
            return html.Div([
                context_div,
                html.Div("No allergen or interaction warnings found for this product.", 
                    style={
                        'backgroundColor': '#d4edda',
                        'border': '1px solid #c3e6cb',
                        'borderRadius': '4px',
                        'padding': '12px',
                        'color': '#155724',
                        'margin': '12px 0'
                    }
                )
            ])

        return html.Div([
            context_div,
            html.Div([
                html.Div(warnings, style={'marginBottom': '8px'}),
                html.Div(interaction_warnings)
            ], style={
                'padding': '12px',
                'backgroundColor': '#fff',
                'borderRadius': '8px',
                'marginBottom': '8px',
                'overflowY': 'auto',
                'maxHeight': '70vh'
            })
        ])

    elif analysis_type == 'details_functions':
        # Generate ingredient details and functions table
        product_brand = product_data_row.get('Brand', 'N/A')
        descriptive_table_div = html.Div([
            html.H5(f"Ingredients Details for {selected_product_name}", className="figure-header"),
            html.P([
                "This table provides detailed information about each ingredient in the product, including its rating, functions, and benefits. ",
                "Hover over column headers for more information about each category."
            ], style={'marginBottom': '15px', 'color': '#666'})
        ])
        paula_details_list = parse_paula_details(product_data_row.get('paula_ingredient_details'))
        raw_ingredients = split_ingredients(product_data_row['Ingredients'])
        
        if paula_details_list and isinstance(paula_details_list, list) and len(paula_details_list) > 0:
            # Create table data from Paula's Choice details
            table_data = []
            for detail in paula_details_list:
                if not isinstance(detail, dict):
                    continue
                ing_name = detail.get('ingredient_name', detail.get('name', 'N/A'))
                rating = detail.get('rating', 'N/A')
                functions = detail.get('functions', '')
                benefits = detail.get('benefits', [])
                if isinstance(benefits, str):
                    benefits = [b.strip() for b in benefits.split(';;') if b.strip()]
                elif isinstance(benefits, list):
                    benefits = [b for b in benefits if b]
                desc = detail.get('description', '')
                table_data.append({
                    'Ingredient': ing_name.capitalize(),
                    'Rating': rating,
                    'Functions': functions,
                    'Benefits': ', '.join(benefits) if benefits else 'N/A',
                    'Description': (desc[:150] + '...' if len(desc)>150 else desc) if desc else 'N/A'
                })
            
            # Create and style the data table
            descriptive_table_div.children.append(dash_table.DataTable(
                columns=[{"name": k, "id": k} for k in table_data[0].keys()],
                data=table_data,
                style_cell={
                    'textAlign': 'left',
                    'fontFamily': 'Nunito Sans, sans-serif',
                    'fontSize': '12px',
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'overflow': 'flex',
                    'padding': '4px 6px'
                },
                style_header={
                    'fontWeight': 'bold',
                    'fontFamily': 'Poppins, sans-serif',
                    'fontSize': '13px',
                    'backgroundColor': 'rgba(0,0,0,0.03)',
                    'position': 'sticky',
                    'top': 0
                },
                style_cell_conditional=[
                    {'if': {'column_id': 'Rating'}, 'minWidth': '60px', 'maxWidth': '120px'}
                ],
                tooltip_header={
                    'Ingredient': 'The ingredient name as listed in the product.',
                    'Rating': "Paula's Choice rating (1-5).",
                    'Functions': 'Functional roles or categories for the ingredient.',
                    'Benefits': 'Key benefits or effects of the ingredient.',
                    'Description': 'A brief description of the ingredient.'
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(0,0,0,0.02)'}
                ],
                style_table={
                    'padding': '8px',
                    'overflowY': 'auto',
                    'overflowX': 'auto',
                    'boxSizing': 'border-box'
                },
                fixed_rows={'headers': True},
                page_size=25
            ))
        else:
            # Fallback to raw ingredient list if no Paula's Choice details available
            if raw_ingredients:
                descriptive_table_div.children.append(
                    html.Div(
                        ', '.join(raw_ingredients),
                        style={
                            'backgroundColor': '#f8d7da',
                            'color': '#222',
                            'padding': '16px',
                            'borderRadius': '8px',
                            'margin': '8px',
                            'fontSize': '16px',
                            'border': '1.5px solid #e09ca0'
                        }
                    )
                )
            else:
                descriptive_table_div.children.append(html.P("No ingredient details extracted."))
        return html.Div([
            html.Div(descriptive_table_div, className="content-card", style={'marginBottom':'12px'})
        ])

    elif analysis_type == 'composition':
        # Generate formulation profile sunburst plot
        paula_details_list = parse_paula_details(product_data_row.get('paula_ingredient_details'))
        sunburst_div = html.Div([html.H5(f"Formulation Profile for {selected_product_name}", className="figure-header")])
        
        if paula_details_list:
            # Prepare data for sunburst plot
            sunburst_data = []
            for detail in paula_details_list:
                ing_name = detail.get('ingredient_name', detail.get('name', 'N/A'))
                parent_cats = detail.get('category', detail.get('categories', []))
                if isinstance(parent_cats, str):
                    parent_cats = [c.strip() for c in parent_cats.split(',') if c.strip()]
                if isinstance(parent_cats, list) and parent_cats:
                    for cat in parent_cats:
                        sunburst_data.append({'category': cat, 'ingredient': ing_name.capitalize(), 'value': 1})
            
            if sunburst_data:
                # Create and style sunburst plot
                sunburst_df = pd.DataFrame(sunburst_data)
                fig = px.sunburst(
                    sunburst_df,
                    path=['category', 'ingredient'],
                    values='value',
                    title='Ingredient Category Sunburst'
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_family="Nunito Sans, sans-serif",
                    font_size=14,
                    height=700,
                    width=700,
                    hoverlabel=dict(
                        bgcolor='white',
                        font_size=14,
                        font_family='Nunito Sans, sans-serif',
                        align='left'
                    )
                )
                sunburst_div.children.append(
                    html.Div(
                        dcc.Graph(
                            figure=fig,
                            style={'width': '100%', 'height': '100%'},
                            config={'responsive': True, 'displayModeBar': True, 'displaylogo': False}
                        ),
                        style={'width': '100%', 'height': '700px'}
                    )
                )
            else:
                sunburst_div.children.append(html.P("No ingredient categories found for sunburst visualization."))
        else:
            sunburst_div.children.append(html.P("No ingredient details available for sunburst visualization."))
        return html.Div([
            html.Div(sunburst_div, className="content-card", style={'marginBottom':'20px', 'height': 'fit-content'})
        ])
    return html.P(f"Analysis type '{analysis_type}' selected. Content to be built.")

@app.callback(
    Output('allergen-dropdown-container', 'style'),
    Input('ia-analysis-type-selector', 'value')
)
def toggle_allergen_dropdown(analysis_type):
    """
    Toggle the visibility of the allergen dropdown based on analysis type.
    
    Args:
        analysis_type (str): The selected analysis type
        
    Returns:
        dict: Style dictionary for the dropdown container
    """
    if analysis_type == 'allergens_interactions':
        return {'display': 'block'}
    return {'display': 'none'}

# Add callback to show/hide the distribution group selector
@app.callback(
    Output('price-distribution-group-container', 'style'),
    Input('price-review-plot-type', 'value')
)
def toggle_distribution_group(plot_type):
    """
    Toggle the visibility of the distribution group selector based on plot type.
    
    Args:
        plot_type (str): The selected plot type
        
    Returns:
        dict: Style dictionary for the distribution group container
    """
    if plot_type == 'box':
        return {'display': 'block', 'marginBottom': '20px'}
    return {'display': 'none'}

if __name__ == '__main__':
    app.run(debug=True)