import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def plot_zscore_ld(zscores, ld, threshold, use_chisq=False, legend_loc="upper_right"):
    # Process inputs to match original dataframe structure
    if use_chisq:
        y_vals = zscores.cpu().detach().numpy()**2
        threshold = threshold**2
    else:
        y_vals = zscores.cpu().detach().numpy()
        
    ld_vals = np.array(ld)
    df = pd.DataFrame({'x': np.arange(len(zscores)), 'y': y_vals, 'ld': ld_vals})

    # Create a conditional column for easier plotting (original logic)
    df['group'] = np.where(df['y'].abs() > threshold, 'y > threshold', 'y <= threshold')

    # Create a separate DataFrame for only the points we want to circle
    df_highlight = df[df['group'] == 'y > threshold']

    # --- Setup Categorical Coloring ---
    # 5 High-contrast colors
    colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#e41a1c']
    cmap = mcolors.ListedColormap(colors)
    # Define 5 bins between 0 and 1
    bounds = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Original figure size
    plt.figure(figsize=(6, 3))
    ax = plt.gca()

    # Main scatter plot (using plt.scatter for colorbar compatibility)
    sc = ax.scatter(
        df['x'], 
        df['y'], 
        c=df['ld'], 
        cmap=cmap, 
        norm=norm, 
        edgecolor='none',
        s=20 # Default seaborn size equivalent
    )

    # On the same axes, plot the "circles" around the highlighted points (Original Specs)
    sns.scatterplot(
        data=df_highlight,
        x='x',
        y='y',
        ax=ax,
        s=150,             # Original size
        facecolors='none',   # Original fill
        edgecolor='red',     # Original color
        linewidth=1.5,
        legend=False
    )

    # Original text labeling logic
    for index, row in df_highlight.iterrows():
        ax.text(
            row['x'] + 0.5,  # X-coordinate for the text (slight offset)
            row['y'],        # Y-coordinate for the text
            int(row['x']),   # The text to display (the x-value)
            horizontalalignment='left',
            size='small',
            color='black'
        )

    # --- Categorical Colorbar in Top Right ---
    axins = inset_axes(ax, width="30%", height="8%", loc=legend_loc, borderpad=1)
    # Ticks centered in the bins [0.1, 0.3, 0.5, 0.7, 0.9]
    cbar = plt.colorbar(
        sc, 
        cax=axins, 
        orientation='horizontal', 
        ticks=bounds
    )
    cbar.ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])    
    cbar.ax.tick_params(labelsize=7)

    # --- Original Final Touches ---
    plt.sca(ax) # Switch back to main axis for labels
    plt.xlabel('Variant')
    if use_chisq:
        plt.ylabel(r'$\chi^2$')
    else:
        plt.ylabel("z-score")
def plot_priors(priors, index):
    df = pd.DataFrame({'x': np.arange(len(priors)), 'y': priors.cpu().detach().numpy()})

    # Create a conditional column for easier plotting
    df['group'] = np.where(np.isin(np.arange(len(priors)), index), 'y > 0.5', 'y <= 0.5')

    # Create a separate DataFrame for only the points we want to circle
    df_highlight = df[df['group'] == 'y > 0.5']

    plt.figure(figsize=(6, 3))
    ax = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='group',
        palette={'y <= 0.5': 'blue', 'y > 0.5': 'red'},
        legend=False
    )

    # On the same axes, plot the "circles" around the highlighted points
    sns.scatterplot(
        data=df_highlight,
        x='x',
        y='y',
        ax=ax,
        s=150,             # Make the marker size larger
        facecolors='none',   # Make the fill transparent
        edgecolor='red',     # Set the circle outline color
        linewidth=1.5,
        legend=False
    )

    for index, row in df_highlight.iterrows():
        ax.text(
            row['x'] + 0.5,  # X-coordinate for the text (slight offset)
            row['y'],        # Y-coordinate for the text
            int(row['x']),   # The text to display (the x-value)
            horizontalalignment='left',
            size='small',
            color='black'
        )

    # --- 4. Final Touches ---
    plt.xlabel('Variant')
    plt.ylabel('PIP')
    plt.ylim(0,1.1)


def plot_zscore(zscores, threshold):
    df = pd.DataFrame({'x': np.arange(len(zscores)), 'y': zscores.cpu().detach().numpy()})

    # Create a conditional column for easier plotting
    df['group'] = np.where(df['y'].abs() > threshold, 'y > threshold', 'y <= threshold')

    # Create a separate DataFrame for only the points we want to circle
    df_highlight = df[df['group'] == 'y > threshold']

    plt.figure(figsize=(6, 3))
    ax = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='group',
        palette={'y <= threshold': 'blue', 'y > threshold': 'red'},
        legend=False
    )

    # On the same axes, plot the "circles" around the highlighted points
    sns.scatterplot(
        data=df_highlight,
        x='x',
        y='y',
        ax=ax,
        s=150,             # Make the marker size larger
        facecolors='none',   # Make the fill transparent
        edgecolor='red',     # Set the circle outline color
        linewidth=1.5,
        legend=False
    )

    for index, row in df_highlight.iterrows():
        ax.text(
            row['x'] + 0.5,  # X-coordinate for the text (slight offset)
            row['y'],        # Y-coordinate for the text
            int(row['x']),   # The text to display (the x-value)
            horizontalalignment='left',
            size='small',
            color='black'
        )

    # --- 4. Final Touches ---
    plt.xlabel('Variant')
    plt.ylabel('z-score')

def plot_finemapping(pip):
    df = pd.DataFrame({'x': np.arange(len(pip)), 'y': pip.cpu().detach().numpy()})

    # Create a conditional column for easier plotting
    df['group'] = np.where(df['y'] > 0.5, 'y > 0.5', 'y <= 0.5')

    # Create a separate DataFrame for only the points we want to circle
    df_highlight = df[df['group'] == 'y > 0.5']

    plt.figure(figsize=(6, 3))
    ax = sns.scatterplot(
        data=df,
        x='x',
        y='y',
        hue='group',
        palette={'y <= 0.5': 'blue', 'y > 0.5': 'red'},
        legend=False
    )

    # On the same axes, plot the "circles" around the highlighted points
    sns.scatterplot(
        data=df_highlight,
        x='x',
        y='y',
        ax=ax,
        s=150,             # Make the marker size larger
        facecolors='none',   # Make the fill transparent
        edgecolor='red',     # Set the circle outline color
        linewidth=1.5,
        legend=False
    )

    for index, row in df_highlight.iterrows():
        ax.text(
            row['x'] + 0.5,  # X-coordinate for the text (slight offset)
            row['y'],        # Y-coordinate for the text
            int(row['x']),   # The text to display (the x-value)
            horizontalalignment='left',
            size='small',
            color='black'
        )

    # --- 4. Final Touches ---
    plt.xlabel('Variant')
    plt.ylabel('PIP')
    #plt.ylim(0,1.1)

def plot_ld_score(pip, ld_score): 
    pip = pip.cpu().detach().numpy()
    ld_score = ld_score.cpu().detach().numpy()
    percentiles = stats.rankdata(ld_score) / len(ld_score)

    fig, ax = plt.subplots(figsize=(6, 3))
    x = np.arange(len(pip))
    scatter = ax.scatter(x, pip, c=percentiles, cmap='viridis', vmin=0, vmax=1, alpha=0.8)
    for i in range(len(x)):
        if pip[i] > 0.5:
            label = f"{percentiles[i]:.2f}"
            ax.text(x[i] + 0.01, pip[i] + 0.01, label, fontsize=8, alpha=0.7)
        
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("LD Score Percentile", rotation=270, labelpad=15)

    # --- 4. Add Labels, Title, and Grid ---
    ax.set_xlabel("Variant")
    ax.set_ylabel("PIP")