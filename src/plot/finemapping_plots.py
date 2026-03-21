import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from scipy.stats import norm, chi2

def plot_manhattan(sum_stats, ld=None, mode="zscore", p_cutoff=5e-8, legend_loc="upper right"):
    if mode == "zscore":
        y_vals = sum_stats
        threshold = norm.ppf(1 - p_cutoff / 2)
        y_label = "z-score"
    elif mode == "chisq":
        y_vals = sum_stats**2
        threshold = chi2.ppf(1 - p_cutoff, df=1)
        y_label = r'$\chi^2$'
    elif mode == "pval":
        p_vals = 2 * norm.sf(np.abs(sum_stats))
        y_vals = -np.log10(np.clip(p_vals, 1e-300, 1))
        threshold = -np.log10(p_cutoff)
        y_label = r'$-\log_{10}(p)$'
    else:
        raise ValueError("mode must be 'zscore', 'chisq', or 'pval'")

    df = pd.DataFrame({'x': np.arange(len(y_vals)), 'y': y_vals})
    
    if ld is not None:
        df['ld'] = np.abs(np.array(ld))

    df['group'] = np.where(np.abs(df['y']) > threshold, 'y > threshold', 'y <= threshold')

    colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#e41a1c']
    cmap = mcolors.ListedColormap(colors)
    bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    mnorm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(6, 3))
    ax = plt.gca()

    if ld is not None:
        sc = ax.scatter(
            df['x'], df['y'], c=df['ld'], 
            cmap=cmap, norm=mnorm, edgecolor='none', s=20
        )
        axins = inset_axes(ax, width="30%", height="8%", loc=legend_loc, borderpad=1)
        cbar = plt.colorbar(sc, cax=axins, orientation='horizontal', ticks=bounds)
        cbar.ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])    
        cbar.ax.tick_params(labelsize=7)
        plt.sca(ax) 
    else:
        sns.scatterplot(
            data=df, x='x', y='y', hue='group', 
            palette={'y <= threshold': 'blue', 'y > threshold': 'red'}, 
            legend=False, ax=ax, s=20, edgecolor='none'
        )

    plt.axhline(y=threshold, color='black', linestyle='--', linewidth=1)
    if mode == "zscore":
        plt.axhline(y=-threshold, color='black', linestyle='--', linewidth=1)

    plt.xlabel('Variant')
    plt.ylabel(y_label)

def plot_pip(pip, cs=None, priors=None, show_labels=False):
    df = pd.DataFrame({'x': np.arange(len(pip)), 'y': pip})
    df['cs_id'] = cs.astype(int) if cs is not None else np.zeros(len(pip), dtype=int)
    
    unique_cs = np.unique(df.loc[df['cs_id'] > 0, 'cs_id'])
    cs_colors = sns.color_palette("Dark2", len(unique_cs))
    cs_palette = {int(cid): cs_colors[i] for i, cid in enumerate(unique_cs)}

    plt.figure(figsize=(6, 3))
    ax = plt.gca()

    if priors is not None:
        sc = ax.scatter(
            df['x'], df['y'], 
            c=priors, cmap='RdBu_r', 
            edgecolor='none', s=20
        )
        
        axins = inset_axes(ax,
                           width="3%",  
                           height="100%",
                           loc='lower left',
                           bbox_to_anchor=(1.02, 0., 1, 1),
                           bbox_transform=ax.transAxes,
                           borderpad=0)
        
        cbar = plt.colorbar(sc, cax=axins)
        cbar.ax.tick_params(labelsize=7)
    else:
        ax.scatter(
            df['x'], df['y'], 
            color='black', edgecolor='none', s=20
        )

    df_highlight = df[df['cs_id'] > 0]
    if not df_highlight.empty:
        ax.scatter(
            df_highlight['x'],
            df_highlight['y'],
            s=150,
            facecolors='none',
            edgecolors=[cs_palette[cid] for cid in df_highlight['cs_id']],
            linewidth=1.5
        )

    if show_labels:
        for index, row in df_highlight.iterrows():
            ax.text(
                row['x'] + 0.5,  
                row['y'],        
                int(row['x']),   
                horizontalalignment='left',
                size='small',
                color='black'
            )

    ax.set_xlabel('Variant')
    ax.set_ylabel('PIP')

def plot_priors(priors, index=None, show_labels=False):
    df = pd.DataFrame({'x': np.arange(len(priors)), 'y': priors})
    
    if index is not None:
        idx_list = np.atleast_1d(index)
        df['is_highlight'] = np.isin(df['x'], idx_list)
    else:
        df['is_highlight'] = False

    df_highlight = df[df['is_highlight']]

    cs_colors = sns.color_palette("Dark2", 1)
    highlight_color = cs_colors[0]

    plt.figure(figsize=(6, 3))
    ax = plt.gca()

    ax.scatter(
        df['x'], 
        df['y'], 
        color='black', 
        edgecolor='none', 
        s=20
    )

    if not df_highlight.empty:
        ax.scatter(
            df_highlight['x'],
            df_highlight['y'],
            s=150,
            facecolors='none',
            edgecolors=highlight_color,
            linewidth=1.5
        )

    if show_labels and not df_highlight.empty:
        for _, row in df_highlight.iterrows():
            ax.text(
                row['x'] + 0.5,  
                row['y'],        
                int(row['x']),  
                horizontalalignment='left',
                size='small',
                color='black'
            )

    plt.xlabel('Variant')
    plt.ylabel('Prior Probability')