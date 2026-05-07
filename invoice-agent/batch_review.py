"""
Quick triage of all extractions — flags potential issues.

Run: python batch_review.py
"""
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

print(f"\n{'='*90}")
print(f"{'#':<3} {'File':<32} {'Conf':<6} {'Items':<7} {'Issues':<35}")
print(f"{'='*90}")

issues_summary = {"high_priority": [], "medium": [], "low": []}

# Sort files alphabetically for consistency
all_files = sorted(RESULTS_DIR.glob("*.json"))

for idx, json_file in enumerate(all_files, 1):
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"{idx:<3} {json_file.stem:<32} ERROR: {e}")
        continue
    
    issues = []
    
    # Check 1: Confidence
    conf = data.get("confidence_score") or 0
    if conf < 0.7:
        issues.append("LOW_CONF")
    
    # Check 2: Suspicious invoice number
    inv = str(data.get("invoice_number") or "")
    if " " in inv:
        issues.append("SPACE_IN_INV#")
    if len(inv) < 3:
        issues.append("SHORT_INV#")
    
    # Check 3: Missing critical fields
    if not data.get("invoice_date"):
        issues.append("NO_DATE")
    if not (data.get("vendor") or {}).get("name"):
        issues.append("NO_VENDOR")
    if not data.get("grand_total"):
        issues.append("NO_TOTAL")
    
    # Check 4: Math mismatch (subtotal + tax - discount = grand_total)
    sub = data.get("subtotal") or 0
    tax = data.get("tax_total") or 0
    disc = data.get("discount") or 0
    grand = data.get("grand_total") or 0
    if grand and sub and abs((sub + tax - disc) - grand) > 1:
        issues.append("MATH_OFF")
    
    # Check 5: Line items math (sum should match subtotal)
    items = data.get("line_items") or []
    if items:
        items_sum = sum((item.get("total") or 0) for item in items)
        if sub and abs(items_sum - sub) > 1:
            issues.append("ITEMS_SUM_OFF")
    
    # Check 6: No line items
    if not items:
        issues.append("NO_ITEMS")
    
    # Check 7: Validation warnings present
    warnings = data.get("validation_warnings") or []
    if warnings:
        issues.append(f"WARN({len(warnings)})")
    
    # Check 8: Suspiciously few line items (might indicate missed rows)
    if 1 <= len(items) <= 2:
        issues.append("FEW_ITEMS")
    
    # Check 9: OCR was used (more error-prone)
    metadata = data.get("raw_extraction_metadata") or {}
    if metadata.get("used_ocr"):
        issues.append("OCR")
    
    issue_str = ", ".join(issues) if issues else "✓ Clean"
    
    name = json_file.stem[:30]
    items_count = len(items)
    print(f"{idx:<3} {name:<32} {conf:<6.2f} {items_count:<7} {issue_str}")
    
    # Categorize for summary
    critical = ["LOW_CONF", "MATH_OFF", "NO_TOTAL", "ITEMS_SUM_OFF", "NO_VENDOR"]
    if any(c in issues for c in critical):
        issues_summary["high_priority"].append(json_file.stem)
    elif issues and issues != ["OCR"]:  # OCR alone doesn't make it medium
        issues_summary["medium"].append(json_file.stem)
    else:
        issues_summary["low"].append(json_file.stem)

print(f"\n{'='*90}")
print(f"📊 TRIAGE SUMMARY:")
print(f"{'='*90}")
print(f"\n🔴 HIGH PRIORITY ({len(issues_summary['high_priority'])} files - has critical issues):")
for f in issues_summary["high_priority"]:
    print(f"   • {f}")

print(f"\n🟡 MEDIUM PRIORITY ({len(issues_summary['medium'])} files - has minor issues):")
for f in issues_summary["medium"][:10]:
    print(f"   • {f}")
if len(issues_summary["medium"]) > 10:
    print(f"   ... and {len(issues_summary['medium']) - 10} more")

print(f"\n🟢 LIKELY CLEAN ({len(issues_summary['low'])} files):")
for f in issues_summary["low"][:5]:
    print(f"   • {f}")
if len(issues_summary["low"]) > 5:
    print(f"   ... and {len(issues_summary['low']) - 5} more")

print(f"\n{'='*90}")
print(f"💡 RECOMMENDED NEXT FILE TO REVIEW:")
if issues_summary["high_priority"]:
    next_file = issues_summary["high_priority"][0]
    # Skip files we've already corrected
    already_done = {"Bryant", "ADM"}
    for f in issues_summary["high_priority"]:
        if not any(done in f for done in already_done):
            next_file = f
            break
    print(f"   {next_file}")
    print(f"\n   Step 1: python find_id.py {next_file.split()[0]}")
    print(f"   Step 2: Open data/sample_invoices/{next_file}.pdf")
    print(f"   Step 3: Open results/{next_file}.json")
print(f"{'='*90}\n")

# Issue frequency analysis
print(f"📈 ISSUE FREQUENCY ACROSS ALL FILES:")
issue_counter = {}
for json_file in all_files:
    try:
        data = json.loads(json_file.read_text(encoding="utf-8"))
    except:
        continue
    
    sub = data.get("subtotal") or 0
    tax = data.get("tax_total") or 0
    disc = data.get("discount") or 0
    grand = data.get("grand_total") or 0
    items = data.get("line_items") or []
    items_sum = sum((item.get("total") or 0) for item in items)
    
    if not data.get("invoice_date"):
        issue_counter["NO_DATE"] = issue_counter.get("NO_DATE", 0) + 1
    if not data.get("grand_total"):
        issue_counter["NO_TOTAL"] = issue_counter.get("NO_TOTAL", 0) + 1
    if grand and sub and abs((sub + tax - disc) - grand) > 1:
        issue_counter["MATH_OFF"] = issue_counter.get("MATH_OFF", 0) + 1
    if sub and abs(items_sum - sub) > 1:
        issue_counter["ITEMS_SUM_OFF"] = issue_counter.get("ITEMS_SUM_OFF", 0) + 1
    if not items:
        issue_counter["NO_ITEMS"] = issue_counter.get("NO_ITEMS", 0) + 1
    if (data.get("raw_extraction_metadata") or {}).get("used_ocr"):
        issue_counter["OCR"] = issue_counter.get("OCR", 0) + 1
    if data.get("validation_warnings"):
        issue_counter["HAS_WARNINGS"] = issue_counter.get("HAS_WARNINGS", 0) + 1

for issue, count in sorted(issue_counter.items(), key=lambda x: -x[1]):
    bar = "█" * int(count * 2)
    print(f"   {issue:<20} {count:>3} files  {bar}")

print()