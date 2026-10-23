from datetime import date

import numpy as np
import pandas as pd
import pytest


def frames():
    members = pd.DataFrame(
        {
            "メンバーコード  ###[member_code]###": ["001", "002", "003", "004"],
            "メンバー名  ###[name]###": ["A", "B", "C", "D"],
            "退職年月日  ###[retired_day]###": ["", "2020-01-01", "", "2030-01-01"],
        }
    )
    acquired = pd.DataFrame(
        {
            "メンバーコード": ["001", "001", "001", "002", "004"],
            "力量タイプ  ###[competence_type]###": ["SKILL", "SKILL", "LICENSE", "SKILL", "SKILL"],
            "力量コード": ["01"] * 5,
            "力量名": ["加工", "加工", "資格", "加工", "加工"],
            "レベル": ["1", "3", "", "4", "2"],
        }
    )
    return members, acquired


def load(members=None, acquired=None):
    from talent_graph.data import prepare_data

    m, a = frames()
    return prepare_data(
        m if members is None else members,
        a if acquired is None else acquired,
        as_of=date(2026, 10, 23),
    )


def test_active_members_without_skills_stay_and_types_do_not_collide():
    data = load()
    assert list(data.members.member_code) == ["001", "003", "004"]
    assert list(data.competences.competence_id) == ["LICENSE:01", "SKILL:01"]
    np.testing.assert_array_equal(data.matrix.toarray(), [[1, 3], [0, 0], [0, 2]])
    assert data.quality["duplicate_acquisitions"] == 1
    assert data.quality["excluded_inactive_records"] == 1


def test_csv_reader_preserves_zero_and_literal_na(tmp_path):
    from talent_graph.data import read_csv

    p = tmp_path / "members.csv"
    p.write_text("メンバーコード,メンバー名\n001,NA\n", encoding="utf-8-sig")
    df = read_csv(p)
    assert df.iloc[0, 0] == "001"
    assert df.iloc[0, 1] == "NA"


@pytest.mark.parametrize("value", ["no", "-1", "nan", "inf", ""])
def test_invalid_skill_level_is_rejected(value):
    _, a = frames()
    a.loc[0, "レベル"] = value
    with pytest.raises(ValueError, match="レベル"):
        load(acquired=a)


def test_duplicate_members_are_rejected():
    m, _ = frames()
    m.loc[1, m.columns[0]] = "001"
    with pytest.raises(ValueError, match="重複"):
        load(members=m)


def test_unknown_member_is_rejected():
    _, a = frames()
    a.loc[0, "メンバーコード"] = "999"
    with pytest.raises(ValueError, match="社員マスタ"):
        load(acquired=a)


def test_missing_required_columns_are_actionable():
    with pytest.raises(ValueError, match="member_code"):
        load(members=pd.DataFrame({"wrong": ["x"]}))


def test_unknown_competence_type_is_rejected():
    _, a = frames()
    a.loc[0, a.columns[1]] = "UNKNOWN"
    with pytest.raises(ValueError, match="力量タイプ"):
        load(acquired=a)


def test_master_fills_missing_name():
    from talent_graph.data import prepare_data

    m, a = frames()
    a = a.drop(columns="力量名")
    masters = {
        "SKILL": pd.DataFrame({"力量コード": ["01"], "力量名": ["加工"]}),
        "LICENSE": pd.DataFrame({"資格コード": ["01"], "資格名": ["資格"]}),
    }
    d = prepare_data(m, a, masters, as_of=date(2026, 10, 23))
    assert set(d.competences.competence_name) == {"加工", "資格"}


def test_invalid_date_is_rejected():
    m, _ = frames()
    m.loc[0, m.columns[2]] = "yesterday"
    with pytest.raises(ValueError, match="退職"):
        load(members=m)


def test_japanese_label_takes_priority_over_generic_annotation():
    from talent_graph.data import prepare_data

    m, a = frames()
    a = a.drop(columns="力量名")
    masters = {
        "SKILL": pd.DataFrame(
            {
                "スキルコード  ###[code]###": ["01"],
                "スキル名  ###[name]###": ["加工"],
            }
        ),
        "LICENSE": pd.DataFrame(
            {
                "資格コード  ###[code]###": ["01"],
                "資格名  ###[name]###": ["資格"],
            }
        ),
    }
    result = prepare_data(m, a, masters, as_of=date(2026, 10, 23))
    assert set(result.competences.competence_name) == {"加工", "資格"}


def test_cp932_reading_and_empty_acquisition_header(tmp_path):
    from talent_graph.data import prepare_data, read_csv

    m, a = frames()
    p = tmp_path / "acquired.csv"
    a.to_csv(p, index=False, encoding="cp932")
    assert read_csv(p, "cp932").iloc[0, 2] == "01"
    result = prepare_data(m, a.iloc[:0], as_of=date(2026, 10, 23))
    assert result.matrix.shape == (3, 0)
    assert result.quality["members_without_acquisitions"] == 3
