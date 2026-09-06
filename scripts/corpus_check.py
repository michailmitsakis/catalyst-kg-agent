import sys, json
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

kg = json.loads(Path("data/processed/kg.json").read_text(encoding="utf-8"))
nodes = kg["nodes"]

mats = [n for n in nodes if n.get("type") == "Material"]
props = [n for n in nodes if n.get("type") == "Property"]
eah = {p["mpid"]: p["value"] for p in props if p.get("name") == "energy_above_hull"}

print(f"nodes={len(nodes)} edges={len(kg['links'])} materials={len(mats)}")
print("node types :", Counter(n.get("type") for n in nodes))
print("properties :", Counter(p.get("name") for p in props))
print("elements   :", sorted({n["symbol"] for n in nodes if n.get("type") == "Element"}))
print()

v = sorted(eah.values())
print(f"e_above_hull: min={v[0]:.4f} max={v[-1]:.4f} median={v[len(v)//2]:.4f}")
print(f"  <= 0.05 (Critic approves): {sum(1 for x in v if x <= 0.05)}")
print(f"  >  0.05 (Critic rejects) : {sum(1 for x in v if x > 0.05)}")
print()

formulas = Counter(m["formula_pretty"] for m in mats)
dup = {k: c for k, c in formulas.items() if c > 1}
print(f"distinct compositions: {len(formulas)} / {len(mats)} materials")
print(f"materials sharing a formula: {sum(dup.values())} ({100*sum(dup.values())/len(mats):.0f}%)")
print("largest polymorph groups:", formulas.most_common(6))
print()
print("oxides:", sum(1 for m in mats if "O" in m["elements"]),
      "| non-oxides:", sum(1 for m in mats if "O" not in m["elements"]))
print("TOP_N cap hit?", "YES - raise it" if len(mats) >= 500 else f"no ({len(mats)} < 500)")