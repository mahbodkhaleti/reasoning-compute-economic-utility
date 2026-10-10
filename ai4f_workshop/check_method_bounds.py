"""Exact algebra fixtures for the proposed certificate, not model evaluation.

This validates finite cases independently of saved experimental outputs. It
does not validate extraction, interval containment on real data, or a service.
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import json


def bounds(value, bids, intervals):
    value = F(value)
    bids = tuple(F(b) for b in bids)
    if not intervals or not bids:
        raise ValueError('Empty population or menu')
    if any(b < 0 or b > value for b in bids):
        raise ValueError('Infeasible bid')
    if any(l < 0 or (u is not None and u < l) for l,u in intervals):
        raise ValueError('Invalid interval')
    n = len(intervals)
    lower = {b:(value-b)*sum(u is not None and u <= b for l,u in intervals)/n for b in bids}
    upper = {b:(value-b)*sum(l <= b for l,u in intervals)/n for b in bids}
    certificates = {b:max([F(0)]+[upper[a]-lower[b] for a in bids if a != b]) for b in bids}
    chosen = min(bids, key=lambda b:(certificates[b],b))
    return lower,upper,certificates,chosen


def utilities(value,bids,prices):
    return {F(b):(F(value)-F(b))*sum(p <= b for p in prices)/len(prices) for b in bids}


def sensitivity(value,bids,intervals,ids):
    n=len(intervals)
    return sum(max((F(value)-b)*int(intervals[j][0] <= b and (intervals[j][1] is None or b < intervals[j][1])) for b in bids) for j in ids)/n


def contained(prices,intervals):
    return all(l <= p and (u is None or p <= u) for p,(l,u) in zip(prices,intervals))


def nested(child,parent):
    return all(lc >= lp and (up is None or (uc is not None and uc <= up)) for (lc,uc),(lp,up) in zip(child,parent))


def main():
    bids=(F(0),F(1),F(2),F(4),F(6))
    choices=((F(0),F(0)),(F(0),F(2)),(F(1),F(3)),(F(2),F(5)),(F(4),None),(F(0),None),(F(3),F(3)))
    states=list(product(choices,repeat=2))
    prices=list(product(tuple(F(i,2) for i in range(13)),repeat=2))
    sandwich_checks=0;regret_checks=0;nested_checks=0;sensitivity_checks=0
    for state in states:
        low,high,cert,chosen=bounds(6,bids,state)
        for ids in ((0,),(1,),(0,1)):
            score=sensitivity(6,bids,state,ids)
            spans=[(F(6)-b)*sum(int(state[j][0] <= b and (state[j][1] is None or b < state[j][1])) for j in ids)/len(state) for b in bids]
            assert score >= max(spans) >= 0
            sensitivity_checks += 1
        for ps in prices:
            if not contained(ps,state):
                continue
            us=utilities(6,bids,ps)
            assert all(low[b] <= us[b] <= high[b] for b in bids)
            assert max(us.values())-us[chosen] <= cert[chosen]
            sandwich_checks += len(bids)
            regret_checks += 1
        for child in states:
            if nested(child,state):
                _,_,cc,cb=bounds(6,bids,child)
                assert all(cc[b] <= cert[b] for b in bids)
                assert cc[cb] <= cert[chosen]
                nested_checks += 1
    # Correct point prices make the chosen action utility-optimal with zero bound.
    for ps in prices:
        _,_,cert,chosen=bounds(6,bids,tuple((p,p) for p in ps))
        us=utilities(6,bids,ps)
        assert cert[chosen] == 0 and us[chosen] == max(us.values())
    # A singleton action needs no repair even with unbounded price ambiguity.
    assert bounds(6,(2,),((F(0),None),))[2][F(2)] == 0
    # Negative control: an incorrect certified point can report zero bound
    # while actual construction regret is positive. Containment is essential.
    _,_,cert,chosen=bounds(10,(2,4),((F(2),F(2)),))
    us=utilities(10,(2,4),(F(4),))
    assert cert[chosen] == 0 and max(us.values())-us[chosen] > cert[chosen]
    # Invalid populations, intervals and actions must not be certified.
    bad=((6,(2,),()),(6,(),((F(0),None),)),(6,(7,),((F(0),None),)),(6,(2,),((F(3),F(2)),)))
    for args in bad:
        try:
            bounds(*args)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid fixture accepted')
    report={
        'passed':True,
        'scope':'Exact-rational finite algebra fixtures, not experiments or model/service evaluation',
        'sandwich_checks':sandwich_checks,'conditional_regret_cases':regret_checks,
        'nested_refinement_pairs':nested_checks,
        'sensitivity_bound_checks':sensitivity_checks,
        'edge_cases':['inclusive threshold ties','bid equal to value','unbounded upper prices','singleton menu','exact point intervals','empty population rejected','empty menu rejected','infeasible bid rejected','empty interval rejected'],
        'invalid_containment_negative_control':'Zero reported certificate can fail when a true price lies outside its interval; condition is essential',
        'limitations':['No real extraction or containment validation','No runtime, utility improvement or statistical coverage evidence','Formal proof, rather than finite fixtures, states the general conditional result']
    }
    Path(__file__).with_name('method_math_checks.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
