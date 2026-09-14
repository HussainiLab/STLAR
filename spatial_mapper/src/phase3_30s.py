"""
Re-run Phase 3 with 30-second chunks for better spatial resolution.
Saves to _stlar_spatial/phase3_30s/ cache.
Produces 3 comparison figures.
"""
import os, sys, warnings
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
SPATIAL_SRC = r'C:/Users/EphysLaptop/Documents/code/python/STLAR/spatial_mapper/src'
if SPATIAL_SRC not in sys.path: sys.path.insert(0, SPATIAL_SRC)
warnings.filterwarnings('ignore')

from PyQt5.QtGui import QGuiApplication
_app = QGuiApplication(sys.argv)
from initialize_fMap import initialize_fMap, compute_polar_binned_analysis
from pathlib import Path
from collections import defaultdict
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

DATA      = Path(r'C:/Users/EphysLaptop/Downloads/AD-LFP/Project_Data')
SPATIAL   = DATA / '_stlar_spatial'
OUT_DIR   = DATA / '_stlar_analysis_stlar_speed' / 'figures'
OUT_DIR.mkdir(exist_ok=True)
# Three speed conditions — one cache dir each
SPEED_CONDITIONS = [
    ('all',    0.0,  100.0),   # no gating (original)
    ('rest',   0.0,    5.0),   # < 5 cm/s
    ('active', 5.0,  100.0),   # >= 5 cm/s
]
cache_dirs = {}
for cond, lo, hi in SPEED_CONDITIONS:
    d = SPATIAL / f'phase3_30s_{cond}'
    d.mkdir(exist_ok=True)
    cache_dirs[cond] = d
# Keep legacy name pointing to 'all' for backward compat
cache_dir = cache_dirs['all']

PPM        = 485
CHUNK_SIZE = 30
EQUAL_AREA_R = 1.0/np.sqrt(2)
N_SECTORS    = 8

GROUPS = ['APOE_Group-1','APOE_Group-2','AppTau_Group-2','AppTau_Group-3']
GROUP_SHORT  = {'APOE_Group-1':'APOE-3','APOE_Group-2':'APOE-4',
                'AppTau_Group-2':'Control','AppTau_Group-3':'AppTau'}

ANIMALS = [
    ('APOE_Group-1','A3',   DATA/'Experiment-1/Group-1/A3'),
    ('APOE_Group-1','A6',   DATA/'Experiment-1/Group-1/A6'),
    ('APOE_Group-1','A7',   DATA/'Experiment-1/Group-1/A7'),
    ('APOE_Group-1','A10',  DATA/'Experiment-1/Group-1/A10'),
    ('APOE_Group-2','A4',   DATA/'Experiment-1/Group-2/A4'),
    ('APOE_Group-2','A5',   DATA/'Experiment-1/Group-2/A5'),
    ('APOE_Group-2','A8',   DATA/'Experiment-1/Group-2/A8'),
    ('APOE_Group-2','A9',   DATA/'Experiment-1/Group-2/A9'),
    ('AppTau_Group-2','NON 73-6',   DATA/'Experiment-2/Group-2/NON 73-6'),
    ('AppTau_Group-2','NON 88a-1',  DATA/'Experiment-2/Group-2/NON 88a-1'),
    ('AppTau_Group-2','NON INT-01', DATA/'Experiment-2/Group-2/NON INT-01'),
    ('AppTau_Group-2','NON INT-02', DATA/'Experiment-2/Group-2/NON INT-02'),
    ('AppTau_Group-2','NON INT-03', DATA/'Experiment-2/Group-2/NON INT-03'),
    ('AppTau_Group-3','ANT 119a-6', DATA/'Experiment-2/Group-3/ANT 119a-6'),
    ('AppTau_Group-3','ANT 120-4',  DATA/'Experiment-2/Group-3/ANT 120-4'),
    ('AppTau_Group-3','ANT 133a-4', DATA/'Experiment-2/Group-3/ANT 133a-4'),
    ('AppTau_Group-3','ANT 135a-7', DATA/'Experiment-2/Group-3/ANT 135a-7'),
    ('AppTau_Group-3','ANT 140-4',  DATA/'Experiment-2/Group-3/ANT 140-4'),
]

BANDS_PLOT = ['Delta','Theta','Beta','Low Gamma','High Gamma','Ripple','Fast Ripple']
BAND_CMAPS = {'Delta':'Purples','Theta':'Blues','Beta':'Greens','Low Gamma':'Oranges',
              'High Gamma':'YlOrRd','Ripple':'Reds','Fast Ripple':'RdPu'}

def is_no(name):
    n = name.upper()
    return '-NO-' in n or n.endswith('-NO.EGF') or 'NOODOR' in n or '-NO.' in n

class BatchWorker:
    class MockSignals:
        def __init__(self): self.progress_value=0; self.progress_text=""
        def emit(self, v=None):
            if isinstance(v, str): self.progress_text=v
            else: self.progress_value=v
    def __init__(self, output_dir=None):
        self.output_dir=output_dir; self.progress_messages=[]; self.signals=self.MockSignals()
        class PS:
            def __init__(self, s): self.w=s
            def emit(self, v):
                if isinstance(v, str): self.w.progress_text=v
                else: self.w.progress_value=v
        class TPS:
            def __init__(self, s): self.w=s
            def emit(self, v): self.w.progress_text=v
        self.signals.progress=PS(self.signals)
        self.signals.text_progress=TPS(self.signals)
    def log(self, msg): self.progress_messages.append(msg)

# ── Main loop ─────────────────────────────────────────────────────────────────
# animal_power keyed by (cond, animal)
animal_power = {}

for (group, animal, animal_dir) in ANIMALS:
    if not animal_dir.exists(): continue
    sessions = [(egf, animal_dir/(egf.stem+'.pos'))
                for egf in sorted(animal_dir.glob('*.egf'))
                if is_no(egf.name) and (animal_dir/(egf.stem+'.pos')).exists()]
    if not sessions: continue
    print(f"\n  {animal} ({GROUP_SHORT[group]}) — {len(sessions)} sessions", flush=True)

    for (cond, lo_spd, hi_spd) in SPEED_CONDITIONS:
        c_dir = cache_dirs[cond]
        band_accum = defaultdict(lambda: np.zeros((2,8)))
        occ_accum  = np.zeros((2,8))
        n_sess     = 0

        for egf_path, pos_path in sessions:
            stem  = egf_path.stem
            cache = c_dir / f'{animal}_{stem}_polar30.npz'

            if cache.exists():
                c = np.load(cache, allow_pickle=True)
                for b in BANDS_PLOT:
                    if b in c: band_accum[b] += c[b]
                if 'occ' in c: occ_accum += c['occ']
                n_sess += 1
                continue

            try:
                worker = BatchWorker(output_dir=str(c_dir))
                result = initialize_fMap(worker,
                                         files=[str(pos_path), str(egf_path)],
                                         ppm=PPM, chunk_size=CHUNK_SIZE,
                                         window_type='hann',
                                         low_speed=lo_spd, high_speed=hi_spd)
                polar = next((x for x in result
                              if isinstance(x, dict) and x.get('type')=='polar'), None)
                if polar is None:
                    chunk_pows = result[4]; tracking = result[5]
                    if not tracking: continue
                    px = np.array([x for chunk in tracking[0] for x in chunk])
                    py = np.array([y for chunk in tracking[1] for y in chunk])
                    pt = (np.array([t for chunk in tracking[2] for t in chunk])
                          if len(tracking) > 2 else np.arange(len(px))/50.)
                    polar = compute_polar_binned_analysis(
                        px, py, pt, 50,
                        list(chunk_pows['Ripple']),
                        CHUNK_SIZE, chunk_pows, result[3])

                sd = {'occ': polar['bin_occupancy']}
                for b in polar['bands']:
                    mp = np.nanmean(polar['bin_power_timeseries'][b], axis=2)
                    band_accum[b] += mp
                    sd[b] = mp
                occ_accum += polar['bin_occupancy']
                np.savez(cache, **sd)
                n_sess += 1
                if cond == 'all':
                    print(f"    [{cond}] {stem}: {polar['time_chunks']} chunks @ 30s", flush=True)

            except Exception as e:
                print(f"    [err] [{cond}] {stem}: {e}")

        if n_sess > 0:
            animal_power[(cond, animal)] = {
                'group': group,
                'bands': {b: band_accum[b]/n_sess for b in band_accum},
                'occ':   occ_accum/n_sess,
            }

print(f"\nAnimals processed: {len(set(a for (_,a) in animal_power.keys()))}")

# ── Plot helpers ──────────────────────────────────────────────────────────────
def draw_polar(ax, data_2x8, title, cmap='hot', vmin=None, vmax=None):
    theta_edges = np.linspace(-np.pi, np.pi, N_SECTORS+1)
    valid = data_2x8[~np.isnan(data_2x8)]
    vmin  = np.nanmin(data_2x8) if vmin is None else vmin
    vmax  = np.nanmax(data_2x8) if vmax is None else vmax
    if vmax <= vmin: vmax = vmin + 1
    norm_   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_fn = cm.get_cmap(cmap)
    for ri, (r_in, r_out) in enumerate([(0, EQUAL_AREA_R), (EQUAL_AREA_R, 1.0)]):
        for s in range(N_SECTORS):
            t1, t2  = theta_edges[s], theta_edges[s+1]
            val     = data_2x8[ri, s]
            color   = cmap_fn(norm_(val)) if not np.isnan(val) else '#ddd'
            ths     = np.linspace(t1, t2, 20)
            xs = np.concatenate([[r_in*np.cos(t1)], r_out*np.cos(ths), [r_in*np.cos(t2)]])
            ys = np.concatenate([[r_in*np.sin(t1)], r_out*np.sin(ths), [r_in*np.sin(t2)]])
            ax.fill(xs, ys, color=color, edgecolor='white', linewidth=0.5)
    ax.add_patch(plt.Circle((0,0), 1.0, fill=False, edgecolor='#555', linewidth=1.2))
    ax.add_patch(plt.Circle((0,0), EQUAL_AREA_R, fill=False, edgecolor='#aaa',
                             linewidth=0.6, linestyle='--'))
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(title, fontsize=8, pad=3)
    sm = cm.ScalarMappable(cmap=cmap_fn, norm=norm_)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02, aspect=15)

def grp_avg(group, band, cond='all'):
    maps = [d['bands'].get(band, np.zeros((2,8)))
            for (c,a), d in animal_power.items() if c==cond and d['group']==group]
    return np.nanmean(maps, axis=0) if maps else np.zeros((2,8))

# ── Fig 1: All 7 bands at 30s ─────────────────────────────────────────────────
if animal_power:
    fig, axes = plt.subplots(len(BANDS_PLOT), 4, figsize=(16, len(BANDS_PLOT)*3.2))
    fig.suptitle('LFP Background Power Maps — 30-second chunks\n'
                 'Polar: 2 rings × 8 sectors  |  Group averages',
                 fontsize=11, fontweight='bold')
    for bi, band in enumerate(BANDS_PLOT):
        maps   = {g: grp_avg(g, band) for g in GROUPS}
        vmax_b = max(np.nanmax(m) for m in maps.values())
        for gi, grp in enumerate(GROUPS):
            lbl = GROUP_SHORT[grp] if bi==0 else ''
            draw_polar(axes[bi,gi], maps[grp], lbl,
                       BAND_CMAPS.get(band,'hot'), 0, vmax_b)
        axes[bi,0].text(-1.3, 0, band, va='center', ha='right',
                        fontsize=8, fontweight='bold', rotation=90)
    plt.tight_layout()
    out = OUT_DIR / 'p3_power_maps_30s.png'
    fig.savefig(out, dpi=110, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out.name}")

# ── Fig 2: 60s vs 30s Ripple comparison ──────────────────────────────────────
phase3_60s = SPATIAL / 'phase3'
animal_power_60s = {}
for grp, animal, _ in ANIMALS:
    npzs = list(phase3_60s.glob(f'{animal}_*_polar.npz'))
    if not npzs: continue
    bt = defaultdict(list)
    for npz in npzs:
        try:
            d = np.load(npz, allow_pickle=True)
            for b in BANDS_PLOT:
                if b in d: bt[b].append(np.array(d[b]))
        except: pass
    if bt:
        animal_power_60s[animal] = {
            'group': grp,
            'bands': {b: np.mean(v, axis=0) for b, v in bt.items()},
        }

if animal_power and animal_power_60s:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle('Ripple Band Power: 60s (top) vs 30s (bottom) chunks\n'
                 'Finer chunks → sharper spatial sampling per bin',
                 fontsize=11, fontweight='bold')
    for gi, grp in enumerate(GROUPS):
        maps_60 = [d['bands'].get('Ripple', np.zeros((2,8)))
                   for a, d in animal_power_60s.items() if d['group']==grp]
        maps_30 = [d['bands'].get('Ripple', np.zeros((2,8)))
                   for (c,a), d in animal_power.items() if c=='all' and d['group']==grp]
        m60 = np.nanmean(maps_60, axis=0) if maps_60 else np.zeros((2,8))
        m30 = np.nanmean(maps_30, axis=0) if maps_30 else np.zeros((2,8))
        all_v = np.concatenate([m60[~np.isnan(m60)], m30[~np.isnan(m30)]])
        vmax  = np.nanmax(all_v) if len(all_v) else 1
        draw_polar(axes[0,gi], m60, f"{GROUP_SHORT[grp]} 60s", 'Reds', 0, vmax)
        draw_polar(axes[1,gi], m30, f"{GROUP_SHORT[grp]} 30s", 'Reds', 0, vmax)
    plt.tight_layout()
    out2 = OUT_DIR / 'p3_ripple_60s_vs_30s.png'
    fig.savefig(out2, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out2.name}")

# ── Fig 3: Theta/Ripple and FR/Ripple ratios at 30s ─────────────────────────
if animal_power:
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle('Spectral Ratios — 30-second chunks\n'
                 'Top: Theta/Ripple  |  Bottom: Fast Ripple/Ripple',
                 fontsize=11, fontweight='bold')
    for gi, grp in enumerate(GROUPS):
        th  = grp_avg(grp, 'Theta')
        rp  = grp_avg(grp, 'Ripple')
        fr  = grp_avg(grp, 'Fast Ripple')
        tr  = np.where(rp>0, th/rp, np.nan)
        frr = np.where(rp>0, fr/rp, np.nan)
        draw_polar(axes[0,gi], tr,  GROUP_SHORT[grp], 'RdYlBu_r')
        draw_polar(axes[1,gi], frr, '',               'PuRd')
    axes[0,0].text(-1.3, 0, 'Theta/Ripple', va='center', ha='right',
                   fontsize=8, fontweight='bold', rotation=90)
    axes[1,0].text(-1.3, 0, 'FR/Ripple',    va='center', ha='right',
                   fontsize=8, fontweight='bold', rotation=90)
    plt.tight_layout()
    out3 = OUT_DIR / 'p3_ratios_30s.png'
    fig.savefig(out3, dpi=130, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out3.name}")

# ── Fig 4: Rest vs Active SSM — Ripple and Fast Ripple ───────────────────────
if animal_power:
    for band, cmap, label in [('Ripple','Reds','Ripple'),
                               ('Fast Ripple','RdPu','Fast Ripple')]:
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle(f'{label} Band Power — Behavioral Gating (30s chunks)\n'
                     'Top: All states  |  Middle: Rest (<5 cm/s)  |  Bottom: Active (≥5 cm/s)',
                     fontsize=11, fontweight='bold')
        for gi, grp in enumerate(GROUPS):
            all_maps  = [d['bands'].get(band, np.zeros((2,8)))
                         for (c,a),d in animal_power.items() if c=='all'    and d['group']==grp]
            rest_maps = [d['bands'].get(band, np.zeros((2,8)))
                         for (c,a),d in animal_power.items() if c=='rest'   and d['group']==grp]
            act_maps  = [d['bands'].get(band, np.zeros((2,8)))
                         for (c,a),d in animal_power.items() if c=='active' and d['group']==grp]
            mall  = np.nanmean(all_maps,  axis=0) if all_maps  else np.zeros((2,8))
            mrest = np.nanmean(rest_maps, axis=0) if rest_maps else np.zeros((2,8))
            mact  = np.nanmean(act_maps,  axis=0) if act_maps  else np.zeros((2,8))
            vmax  = max(np.nanmax(mall), np.nanmax(mrest), np.nanmax(mact))
            vmax  = vmax if vmax > 0 else 1
            draw_polar(axes[0,gi], mall,  f"{GROUP_SHORT[grp]}\nAll",    cmap, 0, vmax)
            draw_polar(axes[1,gi], mrest, f"{GROUP_SHORT[grp]}\nRest",   cmap, 0, vmax)
            draw_polar(axes[2,gi], mact,  f"{GROUP_SHORT[grp]}\nActive", cmap, 0, vmax)
        plt.tight_layout()
        tag  = label.lower().replace(' ','_')
        out4 = OUT_DIR / f'p3_{tag}_rest_vs_active_30s.png'
        fig.savefig(out4, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {out4.name}")

print("\nAll done.")

