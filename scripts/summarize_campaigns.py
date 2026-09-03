import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

for cid in ["demo-001", "demo-002", "demo-003"]:
    p = Path(f"agent/journal/{cid}.json")
    if not p.exists():
        continue
    j = json.loads(p.read_text())
    st = j["campaign_state"]
    bt = j["budget_tracker"]
    print("=" * 60)
    print(f"{cid}: status={st['status']}  evaluated={st['n_materials_evaluated']}")
    print(f"  best={st['best_candidate_mpid']}  e_above_hull={j.get('best_candidate_e_above_hull')}")
    print(f"  spent={bt['total_spent']}  actions={bt['actions_by_category']}")
    for e in j.get("logs", []):
        if e.get("event") == "prioritization":
            print(f"  [planner] provenance={e.get('provenance')} ranked={e.get('n_llm_ranked')} dropped={e.get('n_dropped_unknown')}")
            print(f"            reason: {(e.get('reason') or '')[:150]}")
        elif e.get("event") in ("planner_stop", "candidates_exhausted", "budget_exhausted", "resuming_from_kg"):
            print(f"  [{e['event']}] {({k: v for k, v in e.items() if k not in ('event','campaign_id','timestamp')})}")