"""Builds all paper figures.  python3 make_figures.py [fig01 fig06 ...]   (no argument: everything)
Order of preparation (cached data, see manifest.md):   python3 compute_ll.py;  for T in 0.0005 0.001 0.002 0.004: python3 compute_kag.py $T;  python3 compute_misc.py
One function per figure (fig01() ... fig14(), figS1() ... figS6()); each records its data sources in MANIFEST; manifest.md is written at the end."""
import sys, os, json, time
import style, fig_ll_core as ll, fig_kag as kg, fig_ll_overview as ov, fig_misc as ms, fig_summary as sm
def fig01(): ov.fig01()
def fig02(): ms.fig02()
def fig03(): ms.fig03()
def fig04(): ov.fig04()
def fig05(): ov.fig05()
def fig06(): ll.fig06()
def fig07(): ll.fig07()
def fig08(): ll.fig08()
def fig09(): kg.fig09()
def fig10(): kg.fig10()
def fig11(): kg.fig11()
def fig12(): kg.fig12()
def fig13(): kg.fig13()
def fig14(): sm.fig14()
def figS1(): ms.fig_s1()
def figS2(): ms.fig_s2()
def figS3(): ms.fig_s3()
def figS4(): ms.fig_s4()
def figS5(): ms.fig_s5()
def figS6(): ms.fig_s6()
ALL = [f"fig{i:02d}" for i in range(1, 15)] + [f"figS{i}" for i in range(1, 7)]
if __name__ == "__main__":
    todo = sys.argv[1:] or ALL
    for n in todo:
        t = time.time(); globals()[n](); print(n, f"{time.time() - t:.0f}s", flush=True)
    M = {}
    for mod in (ll, kg, ov, ms, sm): M.update(mod.MP)
    json.dump(M, open(f"{style.HERE}/data/manifest_panels.json", "w"), indent=1)
