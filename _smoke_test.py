import httpx, json, sys

def test_geriatric_polypharmacy(c: httpx.Client) -> bool:
    """Warfarin + Aspirin + Metformin + Lisinopril, age 78, sulfa allergy.
    Expected: 1 High warfarin↔aspirin interaction, 3 elderly dosing recs,
    0 sulfa warnings (no sulfa drugs in list), score 100-30-10-15 = 45.0"""
    payload = {
        "drugs": [
            {"name": "Warfarin", "dosage": "5mg", "frequency": "once daily"},
            {"name": "Aspirin", "dosage": "100mg", "frequency": "once daily"},
            {"name": "Metformin", "dosage": "500mg", "frequency": "twice daily"},
            {"name": "Lisinopril", "dosage": "10mg", "frequency": "once daily"},
        ],
        "patient": {
            "age": 78, "weight": 72,
            "conditions": ["Hypertension", "Type 2 Diabetes", "Atrial Fibrillation"],
            "allergies": ["sulfa"],
        },
        "use_ai_analysis": False,
    }
    r = c.post("/analyze", json=payload)
    print(f"\n[TEST 1] Geriatric polypharmacy — status {r.status_code}")
    if r.status_code != 200:
        print(f"  FAIL: body={r.text}"); return False
    d = r.json()

    extracted_names = sorted({x["name"].lower() for x in d["extracted_drugs"]})
    expected_names = {"warfarin", "aspirin", "metformin", "lisinopril"}
    t1_extracted = extracted_names == sorted(expected_names)
    print(f"  extracted_drugs flows through (unique 4): {'PASS' if t1_extracted else 'FAIL'} -> {extracted_names}")

    interactions = d["interactions"]
    wa_high = [i for i in interactions
               if {i["drug1"].lower(), i["drug2"].lower()} == {"warfarin", "aspirin"}
               and i["severity"] == "High"]
    t1_interaction = len(wa_high) == 1 and len(interactions) == 1
    print(f"  1 High warfarin↔aspirin bidirectional interaction: {'PASS' if t1_interaction else 'FAIL'} (n={len(interactions)})")

    dosing = d["dosage_recommendations"]
    elderly = {x["drug_name"].lower() for x in dosing if "elderly" in x["notes"].lower()}
    t1_dosing = {"warfarin", "aspirin", "metformin"}.issubset(elderly)
    print(f"  Elderly dosage recs for warfarin/aspirin/metformin: {'PASS' if t1_dosing else 'FAIL'} -> {sorted(elderly)}")

    t1_allergies = len(d["allergy_warnings"]) == 0
    print(f"  0 sulfa warnings (correct: 0 sulfa-class drugs in list): {'PASS' if t1_allergies else 'FAIL'} -> {d['allergy_warnings']}")

    t1_score = abs(d["safety_score"] - 45.0) < 0.01
    print(f"  safety_score=45.0 (100-30High-10elderly-15polypharm): {'PASS' if t1_score else 'FAIL'} -> score={d['safety_score']}")

    ok = t1_extracted and t1_interaction and t1_dosing and t1_allergies and t1_score
    print(f"  >>> TEST 1: {'PASS' if ok else 'FAIL'} <<<")
    return ok


def test_penicillin_allergy_hit(c: httpx.Client) -> bool:
    """Amoxicillin (penicillin class) + penicillin allergy.
    Expected: 1 High contraindication via ALLERGY_INDEX set intersection."""
    payload = {
        "drugs": [
            {"name": "Amoxicillin", "dosage": "500mg", "frequency": "three times daily"},
        ],
        "patient": {"age": 45, "conditions": [], "allergies": ["penicillin"]},
        "use_ai_analysis": False,
    }
    r = c.post("/analyze", json=payload)
    print(f"\n[TEST 2] Penicillin class cross-allergy — status {r.status_code}")
    if r.status_code != 200:
        print(f"  FAIL: body={r.text}"); return False
    d = r.json()
    warnings = d["allergy_warnings"]
    w = next((x for x in warnings if x["drug_name"].lower() == "amoxicillin"), None)
    t2_hit = w is not None and w["severity"] == "High" and "CONTRAINDICATED" in w["recommendation"]
    print(f"  Set-intersection found amoxicillin <-> penicillin cross-match: {'PASS' if t2_hit else 'FAIL'} -> {w}")
    score_expected = max(100.0 - 45.0, 0.0)  # 55.0 (45 for 1 allergy)
    t2_score = abs(d["safety_score"] - score_expected) < 0.01
    print(f"  safety_score=55.0 (100-45 allergy): {'PASS' if t2_score else 'FAIL'} -> {d['safety_score']}")
    ok = t2_hit and t2_score
    print(f"  >>> TEST 2: {'PASS' if ok else 'FAIL'} <<<")
    return ok


def test_bidirectional_order(c: httpx.Client) -> bool:
    """Same pair as TEST 1 but reversed: aspirin BEFORE warfarin.
    Expected: same High interaction found (frozenset-key index is order-independent)."""
    payload = {
        "drugs": [
            {"name": "Aspirin", "dosage": "81mg"},
            {"name": "Warfarin", "dosage": "2.5mg"},
        ],
        "patient": {"age": 50, "conditions": [], "allergies": []},
        "use_ai_analysis": False,
    }
    r = c.post("/analyze", json=payload)
    print(f"\n[TEST 3] Bidirectional order independence (aspirin,warfarin) — status {r.status_code}")
    if r.status_code != 200:
        print(f"  FAIL: body={r.text}"); return False
    d = r.json()
    interactions = d["interactions"]
    t3 = len(interactions) == 1 and interactions[0]["severity"] == "High"
    print(f"  Interaction found despite reversed list order: {'PASS' if t3 else 'FAIL'} -> {interactions}")
    ok = t3
    print(f"  >>> TEST 3: {'PASS' if ok else 'FAIL'} <<<")
    return ok


def test_dedup_merging(c: httpx.Client) -> bool:
    """Send DUPLICATE drugs (Warfarin twice + Metformin) + medical_text mentioning 'warfarin 5mg'.
    Expected: only 2 unique drugs in extracted_drugs (set-based dedup working)."""
    payload = {
        "drugs": [
            {"name": "Warfarin", "dosage": "5mg"},
            {"name": "warfarin", "dosage": "2.5mg"},  # case-different duplicate
            {"name": "Metformin", "dosage": "500mg"},
        ],
        "medical_text": "Patient on warfarin 5mg daily",
        "patient": {"age": 55, "conditions": [], "allergies": []},
        "use_ai_analysis": False,
    }
    r = c.post("/analyze", json=payload)
    print(f"\n[TEST 4] Set-based union-deduplication — status {r.status_code}")
    if r.status_code != 200:
        print(f"  FAIL: body={r.text}"); return False
    d = r.json()
    names = [x["name"].lower() for x in d["extracted_drugs"]]
    unique = sorted(set(names))
    t4 = len(d["extracted_drugs"]) == 2 and unique == ["metformin", "warfarin"]
    print(f"  3 duplicates + text -> 2 unique drugs: {'PASS' if t4 else 'FAIL'} -> count={len(d['extracted_drugs'])}, names={[x['name'] for x in d['extracted_drugs']]}")
    ok = t4
    print(f"  >>> TEST 4: {'PASS' if ok else 'FAIL'} <<<")
    return ok


if __name__ == "__main__":
    all_ok = True
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=30.0) as c:
        all_ok &= test_geriatric_polypharmacy(c)
        all_ok &= test_penicillin_allergy_hit(c)
        all_ok &= test_bidirectional_order(c)
        all_ok &= test_dedup_merging(c)

    print(f"\n============\nALL TESTS: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)
