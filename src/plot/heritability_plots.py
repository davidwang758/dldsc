import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_h2(df):
    df_sorted = df.sort_values(by=df.columns[1], ascending=False)
    
    sns.set_theme(style="white")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=df_sorted,
        x=df.columns[0],
        y=df.columns[1],
        hue=df.columns[0],
        palette='viridis',
        legend=False,
        ax=ax
    )
    
    sns.despine(ax=ax, left=False, bottom=True, trim=True)
    ax.tick_params(axis='y', left=True) 
    plt.xticks(rotation=90, fontsize=8)
    
    plt.ylabel('Heritability ($h^2$)', fontsize=12)
    plt.xlabel('Traits', fontsize=12)
    
    plt.tight_layout()

def plot_enrichment_binary(df, trait, top_k=None):
    
    if trait not in df.columns:
        print(f"Error: Trait '{trait}' not found in DataFrame columns.")
        return

    df_subset = df[[trait]].reset_index()
    df_sorted = df_subset.sort_values(by=trait, ascending=False)
    
    if top_k is not None:
        df_sorted = df_sorted.head(top_k)
    
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        data=df_sorted,
        x=df_sorted.columns[0], 
        y=df_sorted.columns[1], 
        hue=df_sorted.columns[0],
        palette='viridis',
        ax=ax
    )
    
    if ax.get_legend():
        ax.get_legend().remove()
    
    sns.despine(ax=ax, left=False, bottom=True, trim=True)
    ax.tick_params(axis='y', left=True) 
    plt.xticks(rotation=90, fontsize=8)
    
    plt.ylabel('Enrichment', fontsize=12)
    plt.xlabel('Annotations', )
    
    plt.title(f'{trait}', fontsize=14)
    
    plt.tight_layout()

def plot_enrichment_continuous(df, trait, top_k_groups=None):
    plot_df = df[[trait]].reset_index()
    plot_df.columns = ['full_annot', 'value']
    
    plot_df['prefix'] = plot_df['full_annot'].str.extract(r'(.*)_q')[0]
    plot_df['q_num'] = plot_df['full_annot'].str.extract(r'_q(\d+)')[0].astype(int)
    
    prefixes_with_nan = plot_df[plot_df['value'].isna()]['prefix'].unique()
    
    plot_df = plot_df[~plot_df['prefix'].isin(prefixes_with_nan)]
    
    if plot_df.empty:
        print(f"No valid groups found for trait '{trait}' after removing NaNs.")
        return

    group_max = plot_df.groupby('prefix')['value'].max().sort_values(ascending=False)
    
    if top_k_groups:
        group_max = group_max.head(top_k_groups)
    
    plot_df = plot_df[plot_df['prefix'].isin(group_max.index)]
    
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.barplot(
        data=plot_df,
        x='prefix',
        y='value',
        hue='q_num',
        palette='viridis',
        order=group_max.index,
        ax=ax
    )
    
    sns.despine(ax=ax, left=False, bottom=True, trim=True)
    ax.tick_params(axis='y', left=True)
    plt.xticks(rotation=90, fontsize=8)
    
    plt.ylabel(f'Enrichment', fontsize=12)
    plt.xlabel('Annotations', fontsize=12)
    plt.title(f'{trait}', fontsize=14)
    
    plt.legend(title='Quantile', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()