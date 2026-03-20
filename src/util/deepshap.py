import shap
import torch
import numpy as np
import pandas as pd

def plot_shap_waterfall(model, annot, cols, task, feature):
    background = annot[np.random.choice(annot.shape[0], 100, replace=False),:]
    explainer = shap.DeepExplainer(model.layers, background)
    shap_vals = explainer(annot[feature,:].reshape(1,-1))
    
    target = pd.Series(annot[feature,:].detach().cpu().numpy())
    target.index = cols

    exp = shap.Explanation(shap_vals.values[0][:,task], 
                  explainer.expected_value[task], 
                  data=target, 
                  feature_names=target.index)

    shap.plots.waterfall(exp)