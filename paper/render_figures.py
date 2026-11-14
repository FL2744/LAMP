from pathlib import Path
import json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
def diagram(path,title,lanes,footer):
 fig,ax=plt.subplots(figsize=(10.6,3.55));ax.set_xlim(0,10.6);ax.set_ylim(0,3.55);ax.axis('off')
 ax.text(.12,3.28,title,fontsize=17,weight='bold',va='center',color='#18232b')
 for row,lane in enumerate(lanes):
  y=2.02-row*1.05
  for col,label in enumerate(lane):
   x=.12+col*3.55
   ax.add_patch(FancyBboxPatch((x,y),3.15,.78,boxstyle='round,pad=0.05,rounding_size=0.07',facecolor='#eef3f5' if row==0 else '#f5f5f5',edgecolor='#8c9da6',lw=.85))
   ax.text(x+1.575,y+.39,textwrap.fill(label,26),ha='center',va='center',fontsize=11.4,color='#17242b')
   if col<2 and not (title.startswith('Three scores') or (title.startswith('Objects are') and row==0)):ax.add_patch(FancyArrowPatch((x+3.22,y+.39),(x+3.48,y+.39),arrowstyle='-|>',mutation_scale=13,color='#486570',lw=1.2))
 ax.text(.12,.30,textwrap.fill(footer,102),fontsize=10.6,va='center',color='#374b57')
 fig.savefig(path.with_suffix('.pdf'),bbox_inches='tight',pad_inches=.06)
 fig.savefig(path.with_suffix('.png'),bbox_inches='tight',pad_inches=.06,dpi=200)
 plt.close(fig)

r=Path(__file__).parent
for k,(_,title,lanes,footer,_) in enumerate(json.loads((r/'figure-specifications.json').read_text()),1):
 diagram(r/'figures'/f'illustration-{k:02}',title,lanes,footer)
