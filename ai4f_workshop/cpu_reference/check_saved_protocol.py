"""Read-only protocol compatibility check, never an independent evaluation."""
from pathlib import Path
import hashlib,json
from reference import Record,source_price

ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'saved_public_development_records.json').read_text())
parsed=[r['id'] for r in rows if source_price(Record(r['id'],r['text']),'USD') is not None]
report={'scope':'Read-only compatibility check on public text from reused development families; not new inference or independent scientific evidence',
        'public_records_checked':len(rows),'accepted_by_new_source_grammar':len(parsed),
        'accepted_ids':parsed,'hidden_labels_loaded':False,
        'conclusion':'The old narrative record protocol is unsupported; the reference does not silently reinterpret or oracle-repair it.',
        'input_sha256':hashlib.sha256((ROOT/'saved_public_development_records.json').read_bytes()).hexdigest()}
(ROOT/'saved_protocol_check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
