"""CPU-only reference contract. No model inference or statistical certification.

The source grammar is a declared public data contract, not a semantic verifier
for unrestricted prose. Fractions preserve inclusive thresholds and exact ties.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import re

DECIMAL = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
ID = re.compile(r"[A-Za-z0-9_-]+\Z")
CURRENCY = re.compile(r"[A-Z]{3}\Z")
FIELD = re.compile(r"record_id=(?P<id>[A-Za-z0-9_-]+); price_to_beat=(?P<amount>(?:0|[1-9][0-9]*)(?:\.[0-9]+)?) (?P<currency>[A-Z]{3})\Z")


def number(x):
    if isinstance(x, bool) or isinstance(x, float):
        raise ValueError('Use exact finite decimal strings or rational values')
    if isinstance(x, str) and not DECIMAL.fullmatch(x):
        raise ValueError('Unsupported decimal')
    if not isinstance(x, (str, int, F)):
        raise ValueError('Unsupported number')
    return F(x)


@dataclass(frozen=True)
class Record:
    ident: str
    text: str


@dataclass(frozen=True)
class Manifest:
    records: tuple
    currency: str
    value: F
    bids: tuple

    @classmethod
    def create(cls, records, currency, value, bids):
        if not isinstance(currency, str) or not CURRENCY.fullmatch(currency):
            raise ValueError('Invalid currency contract')
        items = tuple(Record(r['id'], r['text']) for r in records)
        if not items or any(not isinstance(r.ident, str) or not ID.fullmatch(r.ident) or not isinstance(r.text, str) for r in items):
            raise ValueError('Invalid record manifest')
        if len({r.ident for r in items}) != len(items):
            raise ValueError('Duplicate manifest IDs')
        v = number(value); menu = tuple(number(b) for b in bids)
        if not menu or v < 0 or any(b < 0 or b > v for b in menu) or len(set(menu)) != len(menu):
            raise ValueError('Invalid feasible menu')
        return cls(items, currency, v, tuple(sorted(menu)))


def source_price(record, currency):
    """Pure parser comparator; unknown prose never becomes a point price."""
    m = FIELD.fullmatch(record.text)
    if m is None or m['id'] != record.ident or m['currency'] != currency:
        return None
    return number(m['amount']), m.span('amount')


def validate_intervals(manifest, intervals):
    if set(intervals) != {r.ident for r in manifest.records}:
        raise ValueError('Fixed population changed')
    for lo, hi in intervals.values():
        if not isinstance(lo, F) or (hi is not None and not isinstance(hi, F)) or lo < 0 or (hi is not None and hi < lo):
            raise ValueError('Invalid interval')


def bounds(manifest, intervals):
    validate_intervals(manifest, intervals)
    n = len(manifest.records)
    lower = {b: (manifest.value-b)*sum(hi is not None and hi <= b for lo,hi in intervals.values())/n for b in manifest.bids}
    upper = {b: (manifest.value-b)*sum(lo <= b for lo,hi in intervals.values())/n for b in manifest.bids}
    cert = {b: max([F(0)] + [upper[a]-lower[b] for a in manifest.bids if a != b]) for b in manifest.bids}
    chosen = min(manifest.bids, key=lambda b: (cert[b], b))
    return {'lower': lower, 'upper': upper, 'certificate': cert, 'chosen': chosen}


def chunk_priority(manifest, intervals, chunks, costs):
    validate_intervals(manifest, intervals)
    expected = set(intervals)
    all_ids = [i for g in chunks.values() for i in g]
    if not chunks or len(all_ids) != len(set(all_ids)) or set(all_ids) != expected or set(costs) != set(chunks):
        raise ValueError('Chunks must partition fixed population')
    scores = {}
    for chunk, ids in chunks.items():
        if not isinstance(chunk, str) or not ids:
            raise ValueError('Invalid fixed chunk ID')
        cost = number(costs[chunk])
        if cost <= 0:
            raise ValueError('Nonpositive cost estimate')
        scores[chunk] = sum(max((manifest.value-b)*int(intervals[i][0] <= b and (intervals[i][1] is None or b < intervals[i][1])) for b in manifest.bids) for i in ids)/len(expected)
    eligible = [g for g in chunks if scores[g] > 0]
    selected = min(eligible, key=lambda g: (-scores[g]/number(costs[g]),g)) if eligible else None
    return selected, scores


class State:
    def __init__(self, manifest):
        self.manifest = manifest
        self.intervals = {r.ident: (F(0), None) for r in manifest.records}
        self.log = []
        self.manifest_conflicts = []
        for record in manifest.records:
            m = FIELD.fullmatch(record.text)
            if m is not None and (m['currency'] != manifest.currency or m['id'] != record.ident):
                self.manifest_conflicts.append(record.ident)
        self.sources_sha256 = {r.ident: hashlib.sha256(r.text.encode()).hexdigest() for r in manifest.records}

    def admit_batch(self, proposals, expected_ids):
        """Atomic fixed-chunk update; malformed batches cannot delete records."""
        before = dict(self.intervals)
        try:
            known = {r.ident:r for r in self.manifest.records}
            if len(set(expected_ids)) != len(expected_ids) or not expected_ids or not set(expected_ids) <= set(known):
                raise ValueError('Invalid chunk manifest')
            if not isinstance(proposals, list) or len(proposals) != len(expected_ids):
                raise ValueError('Incomplete batch')
            seen = set(); updates = {}
            for p in proposals:
                if not isinstance(p, dict) or set(p) != {'id','start','end','amount','currency'}:
                    raise ValueError('Invalid proposal schema')
                ident = p['id']
                if not isinstance(ident, str) or ident not in expected_ids or ident in seen:
                    raise ValueError('Unknown or duplicate proposal ID')
                seen.add(ident); record = known[ident]
                parsed = source_price(record, self.manifest.currency)
                if parsed is None:
                    raise ValueError('Unsupported source grammar or unit')
                amount, span = parsed
                if any(type(p[k]) is not int for k in ['start','end']) or (p['start'],p['end']) != span:
                    raise ValueError('Stale or unsupported source span')
                if p['currency'] != self.manifest.currency or number(p['amount']) != amount:
                    raise ValueError('Unsupported amount or currency')
                lo,hi = before[ident]
                if amount < lo or (hi is not None and amount > hi):
                    raise ValueError('Nonnested or conflicting update')
                updates[ident] = (amount,amount)
            if seen != set(expected_ids):
                raise ValueError('Missing required ID')
            self.intervals.update(updates)
            self.log.append({'accepted': True,'ids':sorted(seen),'changed':self.intervals != before})
            return True
        except (ValueError, TypeError, KeyError) as exc:
            self.intervals = before
            self.log.append({'accepted':False,'reason':str(exc)})
            return False

    def decision(self, tolerance):
        epsilon = number(tolerance)
        if epsilon < 0:
            raise ValueError('Negative regret tolerance')
        result = bounds(self.manifest,self.intervals)
        result['status'] = 'conditional_certificate' if not self.manifest_conflicts and result['certificate'][result['chosen']] <= epsilon else 'unresolved'
        if self.manifest_conflicts:
            result['reason'] = 'Public source identity/currency conflicts with manifest'
            result['conflicting_record_ids'] = list(self.manifest_conflicts)
        return result


class BudgetLedger:
    """Output-token and call caps; does not measure GPU latency or input tokens.

    Stage counters are supplied by a future runner, not invented here. Unknown
    token use is conservatively charged at the reserved cap and marked unknown.
    """
    def __init__(self, token_cap, call_cap):
        if type(token_cap) is not int or type(call_cap) is not int or min(token_cap,call_cap) < 0:
            raise ValueError('Invalid call budget')
        self.token_cap = token_cap; self.call_cap = call_cap
        self.used = 0; self.calls = []; self.pending = None

    def reserve(self, output_cap):
        if self.pending is not None:
            raise ValueError('A call is already pending')
        if type(output_cap) is not int or output_cap <= 0:
            raise ValueError('Invalid output cap')
        if len(self.calls) >= self.call_cap or output_cap > self.token_cap-self.used:
            return False
        self.pending = output_cap
        self.calls.append({'reserved_output_tokens':output_cap,'status':'pending'})
        return True

    def finish(self, tokens, stage_seconds, failed=False):
        if self.pending is None:
            raise ValueError('No call reserved')
        if tokens is not None and (type(tokens) is not int or not 0 <= tokens <= self.pending):
            raise ValueError('Runner violated configured output cap')
        stages = {k:number(v) for k,v in stage_seconds.items()}
        if any(v < 0 for v in stages.values()):
            raise ValueError('Invalid stage time')
        charged = self.pending if tokens is None else tokens
        self.used += charged
        self.calls[-1].update(status='failed' if failed else 'finished', charged_output_tokens=charged, token_count_known=tokens is not None, stage_seconds={k:str(v) for k,v in stages.items()})
        self.pending = None


@dataclass(frozen=True)
class CallResult:
    proposals: list
    output_tokens: object
    stage_seconds: dict


def run_policy(manifest, chunks, costs, callback, *, tolerance, token_cap,
               initial_output_cap, repair_output_cap, max_repairs,
               max_repairs_per_chunk):
    """Callback-driven CPU orchestration, not a supplied model service.

    callback(stage, records, output_cap) -> CallResult. Initial extraction is
    one call over all records. Every repair call targets one fixed chunk.
    Source verification is charged, including malformed and failed calls.
    """
    from time import perf_counter
    for cap in (initial_output_cap, repair_output_cap):
        if type(cap) is not int or cap <= 0:
            raise ValueError('Invalid configured output cap')
    if type(max_repairs) is not int or type(max_repairs_per_chunk) is not int or min(max_repairs,max_repairs_per_chunk) < 0:
        raise ValueError('Invalid repair allowance')
    state = State(manifest)
    state.decision(tolerance)  # Validates tolerance before making any call.
    chunk_priority(manifest,state.intervals,chunks,costs)
    ledger = BudgetLedger(token_cap,1+max_repairs)
    repairs = {g:0 for g in chunks}; events=[]

    def outcome(reason):
        d = state.decision(tolerance)
        if reason in ('manifest_conflict','initial_cap_unavailable'):
            d['status'] = 'unresolved'
        return {'decision':d,'reason':reason,'intervals':dict(state.intervals),
                'charged_output_tokens':ledger.used,'calls':ledger.calls,
                'repair_counts':repairs,'admission_log':state.log,'events':events,
                'source_sha256':state.sources_sha256,
                'scope':'Conditional construction certificate only; no held-out or semantic guarantee'}

    def call(stage, ids, cap):
        if not ledger.reserve(cap):
            return False
        rows=tuple(r for r in manifest.records if r.ident in ids)
        start=perf_counter()
        try:
            response=callback(stage,rows,cap)
            elapsed=format(perf_counter()-start,'.12f')
            if not isinstance(response,CallResult):
                raise ValueError('Invalid callback response')
            # Reject invalid token/timing counters before changing state.
            if response.output_tokens is not None and (type(response.output_tokens) is not int or not 0 <= response.output_tokens <= cap):
                raise ValueError('Invalid runtime output count')
            stages={k:number(v) for k,v in response.stage_seconds.items()}
            if any(v < 0 for v in stages.values()):
                raise ValueError('Negative stage time')
            if 'verification_cpu_seconds' in stages or 'callback_wall_seconds' in stages:
                raise ValueError('Reserved accounting stage name')
            verify_start=perf_counter()
            accepted=state.admit_batch(response.proposals,ids)
            verification=format(perf_counter()-verify_start,'.12f')
            stages.update(verification_cpu_seconds=verification,callback_wall_seconds=elapsed)
            ledger.finish(response.output_tokens,stages,failed=not accepted)
            events.append({'stage':stage,'ids':list(ids),'admitted':accepted})
        except Exception as exc:
            # Unknown partial work cannot obtain a free retry. Charge the cap.
            elapsed=format(perf_counter()-start,'.12f')
            ledger.finish(None,{'failed_callback_wall_seconds':elapsed},failed=True)
            events.append({'stage':stage,'ids':list(ids),'admitted':False,'error_type':type(exc).__name__})
        return True

    if state.manifest_conflicts:
        return outcome('manifest_conflict')
    ids=[r.ident for r in manifest.records]
    if not call('initial',ids,initial_output_cap):
        return outcome('initial_cap_unavailable')
    for _ in range(max_repairs+1):
        if state.decision(tolerance)['status'] == 'conditional_certificate':
            return outcome('tolerance_met')
        _,scores=chunk_priority(manifest,state.intervals,chunks,costs)
        eligible=[g for g in chunks if scores[g] > 0 and repairs[g] < max_repairs_per_chunk]
        if len(ledger.calls) >= ledger.call_cap or not eligible or repair_output_cap > ledger.token_cap-ledger.used:
            return outcome('repair_or_generation_cap')
        g=min(eligible,key=lambda g:(-scores[g]/number(costs[g]),g))
        repairs[g]+=1
        call('repair',chunks[g],repair_output_cap)
    return outcome('repair_or_generation_cap')
