# Best Pratices in Jupyter Notebook
- [Pattern Directory Structure](#pattern-directory-structure) :file_folder:
- [Pattern Name of notebook](#pattern-name-of-notebook)
- [Import Declaration](#import-declaration)
- [Prepare Principal Directory](#prepare-principal-directory)
- [Graphics Format](#graphics-format) :bar_chart: :chart_with_upwards_trend:
  - [Cell Format](#cell-format)
  - [Matplotlib](#matplotlib)
  - [Seaborn](#seaborn)
- [Auxiliary Code to Hide the Code in Jupyter](#auxiliary-code-to-hide-the-code-in-jupyter)


### Pattern Directory Structure
```
├── setup.py           <- Make this project pip installable with `pip install -e`
│  
├── LICENSE
│
├── README.md          <- The top-level README for developers using this project.
│
├── .gitignore
|
├── .gitatributes
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│  
├── config-environment <- get configurations
│  
├── data/              <- view TAGs
│
├── notebooks/         <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references/        <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports/           <- Generated analysis as HTML, PDF, LX, graphics and figures to be used in reporting.
│
├── src/               <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── make_dataset.py <- Scripts to download or generate data
```
NOTE: this structure is similary with coockiecutter data science

#### Data Folder
- **raw** <- The original, immutable data dump.
- **external** <- Data from third party sources.
- **processed** <- The final, canonical data sets for modeling.


Notebooks are for exploration and communication

### Pattern Name of notebook
format: `<step><description>.ipynb`
<br/>
Example: `0.3-visualize-distributions.ipynb`

### Import Declaration
Organize all imports in same cell.

```python
# Data analysis and data wrangling
import numpy as np
import pandas as pd

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno

# PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# Preprocessing
from sklearn.preprocessing import LabelEncoder

# Machine learning
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# Dataset
from sklearn.datasets import load_iris

# Other
from IPython.display import Image
import configparser
import subprocess
import warnings
import pprint
import time
import os
```

### Prepare Principal Directory
```python
def change_dir_work(end_directory: str='notebooks'):
    # Current path
    curr_dir = os.path.dirname (os.path.realpath ("__file__")) 
    
    if curr_dir.endswith(end_directory):
        os.chdir('..')
        return curr_dir
    
    return f'Current working directory: {curr_dir}'
```

### Cell Format
```python
# Guarantees visualization inside the jupyter
%matplotlib inline

# OPTIONAL: Load the "autoreload" extension so that code can change
%load_ext autoreload

# Format the data os all table (float_format 3)
pd.set_option('display.float_format', '{:.6}'.format)

# Print xxxx rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Supress unnecessary warnings so that presentation looks clean
warnings.filterwarnings('ignore')

# Pretty print
pp = pprint.PrettyPrinter(indent=4)
```

### Graphics Format
It is recommended to use a resource file, such as [this](src/visualization/plot_config.ini). This ensures better standardization of notebooks.
```python
config = configparser.ConfigParser()
config.read('src/visualization/plot_config.ini')

figure_titlesize = config['figure']['figure_titlesize']
figure_figsize_large = int(config['figure']['figure_figsize_large'])
figure_figsize_width = int(config['figure']['figure_figsize_width'])
figure_dpi = int(config['figure']['figure_dpi'])
figure_facecolor = config['figure']['figure_facecolor']
figure_autolayout = bool(config['figure']['figure_autolayout'])
savefig_format = config['figure']['savefig_format']
savefig_dpi = int(config['figure']['savefig_dpi'])

axes_labelsize = int(config['axes']['axes_labelsize'])
axes_titlesize = int(config['axes']['axes_labelsize'])

lines_antialiased = bool(config['line']['lines_antialiased'])
lines_linewidth = int(config['line']['lines_linewidth'])
lines_color = config['line']['lines_color']

hist_bins = int(config['histogram']['hist_bins'])

boxplot_vertical = bool(config['boxplot']['boxplot_vertical'])
boxplot_showmeans = bool(config['boxplot']['boxplot_showmeans'])
boxplot_showbox = bool(config['boxplot']['boxplot_showbox'])
boxplot_flierprops_color = config['boxplot']['boxplot_flierprops_color']
boxplot_flierprops_markerfacecolor = config['boxplot']['boxplot_flierprops_markerfacecolor']
boxplot_flierprops_markersize = int(config['boxplot']['boxplot_flierprops_markersize'])
boxplot_meanprops_linewidth = int(config['boxplot']['boxplot_meanprops_linewidth'])

font_family = config['font']['font_family']
font_size = int(config['font']['font_size'])

legend_loc = config['legend']['legend_loc']
legend_fontsize = int(config['legend']['legend_fontsize'])
```

### Matplotlib
```python
# ===================
# matplotlib rcParams
# ===================
plt.style.use('fivethirtyeight')

# Figure
plt.rcParams['figure.titlesize'] = figure_titlesize
plt.rcParams['figure.figsize'] = [figure_figsize_large, figure_figsize_width] 
plt.rcParams['figure.dpi'] = figure_dpi
plt.rcParams['figure.facecolor'] = figure_facecolor
plt.rcParams['figure.autolayout'] = figure_autolayout

plt.rcParams['savefig.format'] = savefig_format
plt.rcParams['savefig.dpi'] = savefig_dpi

# Axes
plt.rcParams['axes.labelsize'] = axes_labelsize
plt.rcParams['axes.titlesize'] = axes_titlesize

# Lines
plt.rcParams['lines.antialiased'] = lines_antialiased
plt.rcParams['lines.linewidth'] = lines_linewidth
plt.rcParams['lines.color'] = lines_color
 
# Histogram Plots
plt.rcParams['hist.bins'] = hist_bins

# Boxplot
plt.rcParams['boxplot.vertical'] = boxplot_vertical
plt.rcParams['boxplot.showmeans'] = boxplot_showmeans
plt.rcParams['boxplot.showbox'] = boxplot_showbox
plt.rcParams['boxplot.flierprops.color'] = boxplot_flierprops_color
plt.rcParams['boxplot.flierprops.markerfacecolor'] = boxplot_flierprops_markerfacecolor
plt.rcParams['boxplot.flierprops.markersize'] = boxplot_flierprops_markersize
plt.rcParams['boxplot.meanprops.linewidth'] = boxplot_meanprops_linewidth

# Font
plt.rcParams['font.family'] = font_family
plt.rcParams['font.size'] = font_size

# Legend
plt.rcParams['legend.loc'] = legend_loc
plt.rcParams['legend.fontsize'] = legend_fontsize
```

### Seaborn 
```python
# ===================
# Seaborn rcParams
# ===================
rc={'savefig.dpi': 500, 
    'figure.autolayout': True, 
    'figure.figsize': [18, 8], 
    'axes.labelsize': 18,
    'axes.titlesize': 18, 
    'font.size': 15, 
    'lines.linewidth': 1.0, 
    'lines.markersize': 8, 
    'legend.fontsize': 15,
    'xtick.labelsize': 15, 
    'ytick.labelsize': 15}

sns.set(font=font_family,
        style='darkgrid',
        palette='deep',
        color_codes=True,
        rc=rc)
```

### Auxiliary Code to Hide the Code in Jupyter
This code causes the notebook cells to be hidden. For this, the cells must contain the word `# hide_code`. See this [example](https://github.com/brunocampos01/finding-donors/blob/master/notebooks/finding_donors.ipynb).

```javascript
%%html

<script>
code_show = true;

function code_display() {
    if (!code_show) {
        $('div.input').each(function (id) {
            $(this).show();
        });
        $('div.output_prompt').css('opacity', 1);
    } else {
        $('div.input').each(function (id) {
            if (id == 0 || $(this).html().indexOf('hide_code') > -1) {
                $(this).hide();
            }
        });
        $('div.output_prompt').css('opacity', 0);
    }
    ;
    code_show = !code_show;
}

$(document).ready(code_display);
</script>

<form action="javascript: code_display()">
    <input style="color: #0f0c0c; background: LightGray; opacity: 0.8;" \
    type="submit" value="Click to display or hide code cells">
</form>
``` 
