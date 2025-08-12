from . import tst_data_loader as loader
from . import tst_prep as prep
from . import tst_visualizer as vis

from . import resources

from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

font_path = 'C:/Windows/Fonts/gulim.ttc'
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)

def run():
    data = loader.load_raw_data()
    if data is None:
        return None
    # 클렌징
    
    start = datetime.now()
    
    print('cleaning data...', end='')
    data['titles'] = data['titles'].apply(prep.normalize_punct)
    print(f'\tDONE (start: {start}, end: {datetime.now()}, time: {datetime.now() - start}')

    # 토큰화
    
    start = datetime.now()
    
    print('tokenizing data...', end='')
    data['titles'] = data['titles'].apply(prep.custom_tokenize)
    print(f'\tDONE (start: {start}, end: {datetime.now()}, time: {datetime.now() - start}')

    # 히트맵 시각화
    
    start = datetime.now()
    
    print('reshaping data...', end='')
    data = vis.melt_titles(data)

    hm_data = data.pivot_table(
        values='counts',
        index='dates',
        columns='tokens',
        aggfunc='sum',
        fill_value=0
    )
    
    hm_tokens = hm_data.sum().sort_values(ascending=False).iloc[:100].index
    
    fig = plt.figure(figsize=(25, 8))

    ax = sns.heatmap(
        hm_data[hm_tokens]
    )
    
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=15)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    
    plt.savefig('tst_hm.png')

    print(f'\tDONE (start: {start}, end: {datetime.now()}, time: {datetime.now() - start}')
    
    plt.show()
