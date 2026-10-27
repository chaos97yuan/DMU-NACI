from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval import historical_interpolation_causality_audit as a


def flags(year=2022, n=120):
    f=pd.DataFrame({'timestamp':pd.date_range(f'{year}-01-01',periods=n,freq='5min'),
                    'segment_id':1,'row_has_core_missing':0})
    for col in a.FLAGS.values(): f[col]=0
    return f


@pytest.mark.parametrize('year',[2022,2023])
@pytest.mark.parametrize('offset,unsafe',[(1,True),(0,False),(-1,False)])
def test_endpoint_relation(year,offset,unsafe):
    f=flags(year);ts=pd.DatetimeIndex(f.timestamp);origin=ts[71]
    lineage=pd.DataFrame({'missing_timestamp':[ts[70]],'right_endpoint_timestamp':[origin+pd.Timedelta(minutes=offset*5)]})
    assert a.is_future_endpoint_dependent(origin,ts[:72],lineage)==unsafe


def test_causal_ffill_safe():
    f=flags();ts=pd.DatetimeIndex(f.timestamp)
    l=pd.DataFrame({'missing_timestamp':[ts[70]],'right_endpoint_timestamp':[pd.NaT]})
    assert not a.is_future_endpoint_dependent(ts[71],ts[:72],l)


@pytest.mark.parametrize('col',list(a.FLAGS.values()))
def test_multiple_cells_and_raw_used(col):
    f=flags(); f.loc[71:73,col]=1
    l=a.reconstruct_lineage(f)
    assert len(l)==3
    c,d=a.audit_candidates(f,l)
    assert c.future_endpoint_dependent.sum()==3
    assert c.loc[c.future_endpoint_dependent,'retained'].sum()==0
    assert (d.right_endpoint_timestamp==f.timestamp.iloc[74]).all()


def test_gap_qc_separate():
    f=flags();f.loc[71,'power_imputed']=1;f.loc[20,'segment_id']=-1
    c,d=a.audit_candidates(f,a.reconstruct_lineage(f))
    assert c.future_endpoint_dependent.sum()==1
    assert not c.loc[0,'history_qc_eligible']


def test_saved_linear_values_verified():
    f=flags();f.loc[71:72,'power_imputed']=1
    core=pd.DataFrame({'timestamp':f.timestamp})
    for col in a.FLAGS:core[col]=np.arange(len(f),dtype=float)
    assert len(a.reconstruct_lineage(f,core))==2
    core.loc[71,'Active_Power']=1000
    with pytest.raises(ValueError,match='not reproduced'):a.reconstruct_lineage(f,core)


@pytest.mark.parametrize('used,raw,resolved,level,gate',[
    (0,0,True,0,'PASS'),(0,1,True,1,'PASS'),(1,1,True,2,'HOLD'),
    (10,10,True,3,'FAIL'),(0,0,False,4,'HOLD')])
def test_severity(used,raw,resolved,level,gate):
    df=pd.DataFrame({'UsedDependent':[used],'RawDependent':[raw],'UsedRate':[used/10000]})
    result,g=a.severity(df,resolved)
    assert result.startswith(f'LEVEL_{level}_') and g==gate


@pytest.mark.parametrize('split',['train','val','cal'])
def test_actual_membership_selection(split):
    f=flags();f.loc[71,'power_imputed']=1;c,_=a.audit_candidates(f,a.reconstruct_lineage(f))
    actual=c.set_index('forecast_origin_timestamp').loc[[f.timestamp.iloc[72]]]
    assert not actual.future_endpoint_dependent.any()
    assert c.future_endpoint_dependent.sum()==1


def test_determinism():
    f=flags();f.loc[71,'power_imputed']=1
    x=a.audit_candidates(f,a.reconstruct_lineage(f));y=a.audit_candidates(f,a.reconstruct_lineage(f))
    assert all(p.equals(q) for p,q in zip(x,y))
    assert a.rule_definition()==a.rule_definition()


@pytest.mark.parametrize('side',[0,119])
def test_unresolved_endpoint_fails(side):
    f=flags();f.loc[side,'power_imputed']=1
    with pytest.raises(ValueError,match='Unresolved'):a.reconstruct_lineage(f)


def test_missing_grid_fails():
    f=flags().drop(index=15)
    with pytest.raises(ValueError,match='grid'):a.reconstruct_lineage(f)


def test_invalid_flags_fail():
    f=flags();f.loc[20,'ghi_imputed']=2
    with pytest.raises(ValueError,match='Invalid'):a.reconstruct_lineage(f)


def test_no_performance_or_training_calls():
    import ast,inspect
    import src.eval.historical_interpolation_causality_audit as module
    tree=ast.parse(inspect.getsource(module))
    names={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
    assert not names & {'fit','fit_transform','compute_error','conformal_metrics','backward'}


# Release note: the original test suite also contained
# test_target_guard_still_blocks(), which imports the legacy module
# src/eval/n_grouped_external_validation.py (not part of this release); it was
# dropped here -- see CHANGELOG.md.


def test_history_outside_window_safe():
    ts=pd.DatetimeIndex(flags().timestamp)
    l=pd.DataFrame({'missing_timestamp':[ts[0]],'right_endpoint_timestamp':[ts[100]]})
    assert not a.is_future_endpoint_dependent(ts[90],ts[19:91],l)
