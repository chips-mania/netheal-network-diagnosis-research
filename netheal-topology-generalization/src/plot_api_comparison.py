"""Static figures for completed API pilots."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
def main():
    ap=argparse.ArgumentParser();ap.add_argument("run");args=ap.parse_args();out=Path(args.run)
    rows=[json.loads(x) for x in (out/"episodes.jsonl").read_text(encoding="utf8").splitlines()]
    fig,axes=plt.subplots(1,3,figsize=(14,4.6))
    colors={"gpt":"#2463a7","jev":"#d17624"}
    for j,p in enumerate(("gpt","jev")):
        rs=[r for r in rows if r["provider"]==p]
        if not rs:continue
        axes[0].bar(j,100*np.mean([r["exact"] for r in rs]),color=colors[p],width=.55)
        for ax,key in ((axes[1],"wall_seconds"),(axes[2],"estimated_usd")):
            vals=[r[key] for r in rs if r[key] is not None]
            if vals:
                ax.scatter(np.linspace(j-.14,j+.14,len(vals)),vals,c=[colors[p]],alpha=.65,s=18)
                ax.plot([j-.25,j+.25],[np.median(vals)]*2,c="black",lw=2)
    axes[0].set_ylim(0,100);axes[0].set_ylabel("Exact diagnosis (%)")
    axes[1].set_ylabel("End-to-end seconds");axes[1].set_yscale("log")
    axes[2].set_ylabel("Estimated USD / episode");axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xticks([0,1],["GPT-6.1 Sol","Jev 1.13"]);ax.grid(axis="y",alpha=.2)
    fig.suptitle("Paired development pilot: same tasks, observed time and cost")
    fig.text(.5,.01,"Each dot is an episode; black lines are medians. Costs from API usage, not reconciled invoices.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(out/"comparison.png",dpi=180);plt.close(fig)
if __name__=="__main__":main()
