"""Gouda-2.0 pretrained vs instruct benchmarks, with Pythia reference lines.
Run: pip install matplotlib numpy && python gouda_benchmarks_revised.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

MODELS = ['Ricotta\n24M', 'Gruyère\n63M', 'Gouda\n132M', 'Cheddar\n302M']
BENCHMARKS = {
    'HellaSwag':       {'pre': [.283,.312,.326,.327], 'inst': [.282,.307,.328,.330], 'py160': None, 'py410': None},
    'PIQA':            {'pre': [.596,.620,.626,.624], 'inst': [.603,.613,.633,.643], 'py160': .627, 'py410': .668},
    'ARC Easy':        {'pre': [.403,.439,.458,.459], 'inst': [.393,.428,.418,.455], 'py160': .435, 'py410': .521},
    'ARC Challenge':   {'pre': [.226,.242,.255,.257], 'inst': [.230,.236,.270,.272], 'py160': .188, 'py410': .213},
    'SciQ':            {'pre': [.612,.644,.669,.654], 'inst': [.568,.628,.626,.653], 'py160': .741, 'py410': .811},
    'LAMBADA (OpenAI)':{'pre': [.213,.248,.266,.262], 'inst': [.231,.265,.292,.283], 'py160': .328, 'py410': .516},
    'OpenBookQA':      {'pre': [.286,.284,.294,.316], 'inst': [.282,.298,.314,.310], 'py160': None, 'py410': None},
    'BoolQ':           {'pre': [.490,.607,.546,.571], 'inst': [.516,.608,.523,.601], 'py160': None, 'py410': None},
}

BLUE = 'tab:blue'
ORANGE = 'tab:orange'
AVERAGE = 'black'
PY160 = 'tab:green'
PY410 = 'tab:red'

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
})

fig, axes = plt.subplots(4, 2, figsize=(16, 17), constrained_layout=True)
x = np.arange(len(MODELS))
width = .33

for ax, (name, d) in zip(axes.flat, BENCHMARKS.items()):
    pre = np.array(d['pre'], dtype=float)
    inst = np.array(d['inst'], dtype=float)
    average = (pre + inst) / 2

    ax.bar(x - width / 2, pre, width, color=BLUE, label='Pretrained', zorder=2)
    ax.bar(x + width / 2, inst, width, color=ORANGE, label='Instruct', zorder=2)
    ax.plot(x, average, color=AVERAGE, marker='o', markersize=5,
            linewidth=1.8, zorder=4, label='Group mean')

    for key, color, label in [('py160', PY160, 'Pythia-160M'),
                              ('py410', PY410, 'Pythia-410M')]:
        if d[key] is not None:
            ax.axhline(d[key], color=color, linestyle='--', linewidth=1.5,
                       alpha=.9, zorder=1)
            ax.text(.99, d[key], f' {label}: {d[key]:.3f}',
                    transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                    fontsize=8, color=color,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.82, pad=.5))

    values = np.r_[pre, inst, average]
    refs = [d[k] for k in ('py160', 'py410') if d[k] is not None]
    upper = max([*values, *refs])
    lower = min([*values, *refs])
    # Use a zero baseline for bar charts to avoid exaggerating differences.
    ax.set_ylim(0, min(1.04, upper * 1.20 + .025))

    for xpos, vals in ((x - width / 2, pre), (x + width / 2, inst)):
        for xi, yi in zip(xpos, vals):
            ax.annotate(f'{yi:.3f}', (xi, yi), xytext=(0, 3),
                        textcoords='offset points', ha='center', va='bottom',
                        fontsize=7.5)

    ax.set_title(name, fontsize=12, fontweight='bold', pad=9)
    ax.set_xticks(x, MODELS)
    ax.set_ylabel('Accuracy')
    ax.grid(axis='y', alpha=.18)
    ax.set_axisbelow(True)
    if d['py160'] is None and d['py410'] is None:
        ax.text(.02, .96, 'Pythia-160M: N/A  |  Pythia-410M: N/A',
                transform=ax.transAxes, va='top', ha='left', fontsize=8, color='0.45')

legend = [
    Patch(color=BLUE, label='Gouda pretrained'),
    Patch(color=ORANGE, label='Gouda instruct'),
    Line2D([0], [0], color=AVERAGE, marker='o', label='Mean of pretrained + instruct'),
    Line2D([0], [0], color=PY160, ls='--', label='Pythia-160M (pretrained)'),
    Line2D([0], [0], color=PY410, ls='--', label='Pythia-410M (pretrained)'),
]
fig.suptitle('Gouda-2.0 | Benchmark comparison', fontsize=19, fontweight='bold')
fig.legend(handles=legend, loc='outside upper center', bbox_to_anchor=(.5, .978),
           ncol=5, frameon=False, fontsize=9)
fig.supxlabel('Pythia reference values: published zero-shot results; N/A where not reported.\n'
              'Cross-paper comparisons may differ in evaluation harness and settings.', fontsize=9)
fig.savefig('gouda_benchmark_comparison.png', dpi=220, bbox_inches='tight')
plt.close(fig)
print('Saved gouda_benchmark_comparison.png')
