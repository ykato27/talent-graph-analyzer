"""Skillnote CSVの読み取りと検証。コードは常に文字列として保持する。"""

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


class DataValidationError(ValueError):
    """入力データを修正する必要がある。"""


ALIASES = {
    "member_code": ("member_code", "メンバーコード", "社員コード"),
    "name": ("name", "member_name", "メンバー名", "社員名"),
    "retired_day": ("retired_day", "退職年月日", "退職日"),
    "enter_day": ("enter_day", "入社年月日", "入社日"),
    "job_grade": ("job_grade", "職能・等級", "等級"),
    "job_position": ("job_position", "役職"),
    "competence_type": ("competence_type", "力量タイプ", "力量種別"),
    "competence_code": (
        "competence_code",
        "力量コード",
        "スキルコード",
        "資格コード",
        "教育コード",
    ),
    "competence_name": ("competence_name", "力量名", "スキル名", "資格名", "教育名"),
    "level": ("level", "レベル", "習得レベル"),
}
TYPES = {
    "SKILL": "SKILL",
    "スキル": "SKILL",
    "EDUCATION": "EDUCATION",
    "教育": "EDUCATION",
    "LICENSE": "LICENSE",
    "資格": "LICENSE",
}


@dataclass
class SkillnoteData:
    members: pd.DataFrame
    competences: pd.DataFrame
    matrix: csr_matrix
    quality: dict
    as_of: date


def read_csv(path: Path, encoding: str = "utf-8-sig") -> pd.DataFrame:
    """数値コード・文字列NA・空欄を変更せずに読む。"""
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeError) as exc:
        raise DataValidationError(
            f"{Path(path).name}: CSV形式または文字コードを確認してください"
        ) from exc


def normalize_columns(frame: pd.DataFrame, required: tuple[str, ...]) -> pd.DataFrame:
    """注釈付きエクスポート列と通常の日本語・英語列を同じ列名へ変換する。"""
    frame = frame.copy()
    renames = {}
    for column in frame.columns:
        text = str(column).strip().lstrip("\ufeff")
        annotation = re.search(r"###\[([^\]]+)\]###", text)
        label = text.split("###")[0].strip()
        candidate = annotation.group(1) if annotation else label
        # 日本語の表示名が列の意味を特定する場合は、汎用的な[name]注釈より優先する。
        for key in (label, candidate):
            matched = next(
                (canonical for canonical, names in ALIASES.items() if key in names), None
            )
            if matched:
                renames[column] = matched
                break
    frame = frame.rename(columns=renames)
    if frame.columns.duplicated().any():
        raise DataValidationError("同じ意味の列が重複しています。CSVの列を確認してください")
    missing = set(required) - set(frame.columns)
    if missing:
        raise DataValidationError(f"必須列が不足しています: {', '.join(sorted(missing))}")
    for col in frame.columns:
        frame[col] = frame[col].fillna("").astype(str).str.strip()
    return frame


def _dates(series: pd.Series, label: str) -> pd.Series:
    result = pd.to_datetime(series.replace("", None), errors="coerce", format="mixed")
    if ((series != "") & result.isna()).any():
        raise DataValidationError(
            f"{label}に無効な日付があります。YYYY-MM-DD形式で指定してください"
        )
    return result


def prepare_data(
    member_frame: pd.DataFrame,
    acquired_frame: pd.DataFrame,
    masters: dict[str, pd.DataFrame] | None = None,
    *,
    as_of: date | None = None,
) -> SkillnoteData:
    """在籍社員×力量の疎行列。重複習得は最大レベル、資格・教育は保有=1。"""
    as_of = as_of or date.today()
    members = normalize_columns(member_frame, ("member_code", "name"))
    acquired = normalize_columns(
        acquired_frame, ("member_code", "competence_type", "competence_code")
    )
    if members.empty:
        raise DataValidationError("社員マスタが空です")
    for df, cols in [
        (members, ["member_code", "name"]),
        (acquired, ["member_code", "competence_type", "competence_code"]),
    ]:
        if df[cols].eq("").any().any():
            raise DataValidationError(f"必須項目が空欄です: {', '.join(cols)}")
    if members.member_code.duplicated().any():
        raise DataValidationError("社員マスタのメンバーコードが重複しています")
    unknown_members = ~acquired.member_code.isin(members.member_code)
    if unknown_members.any():
        raise DataValidationError(
            f"社員マスタにないメンバーコードが{unknown_members.sum()}行あります"
        )
    for col in ("retired_day", "enter_day", "job_grade", "job_position"):
        if col not in members:
            members[col] = ""
    retired = _dates(members.retired_day, "退職年月日")
    entered = _dates(members.enter_day, "入社年月日")
    active = (retired.isna() | (retired > pd.Timestamp(as_of))) & (
        entered.isna() | (entered <= pd.Timestamp(as_of))
    )
    total_members = len(members)
    members = members.loc[active].reset_index(drop=True)
    if members.empty:
        raise DataValidationError("基準日時点の在籍社員がいません")
    types = acquired.competence_type.map(lambda x: TYPES.get(x.upper(), TYPES.get(x)))
    if types.isna().any():
        raise DataValidationError("力量タイプはSKILL / EDUCATION / LICENSEを指定してください")
    acquired["competence_type"] = types
    if "level" not in acquired:
        acquired["level"] = ""
    skill_rows = acquired.competence_type.eq("SKILL")
    levels = pd.to_numeric(acquired.level, errors="coerce")
    if (skill_rows & (~np.isfinite(levels) | (levels < 0))).any():
        raise DataValidationError("スキルのレベルは0以上の有限数値が必須です")
    acquired["value"] = np.where(skill_rows, levels, 1.0)
    if "competence_name" not in acquired:
        acquired["competence_name"] = ""
    for kind, master in (masters or {}).items():
        master = normalize_columns(master, ("competence_code", "competence_name"))
        if master.competence_code.duplicated().any():
            raise DataValidationError(f"{kind}マスタの力量コードが重複しています")
        names = master.set_index("competence_code").competence_name
        mask = acquired.competence_type.eq(kind) & acquired.competence_name.eq("")
        acquired.loc[mask, "competence_name"] = (
            acquired.loc[mask, "competence_code"].map(names).fillna("")
        )
    unnamed = acquired.competence_name.eq("")
    acquired.loc[unnamed, "competence_name"] = acquired.loc[unnamed, "competence_code"]
    in_scope = acquired.member_code.isin(members.member_code)
    excluded = int((~in_scope).sum())
    acquired = acquired.loc[in_scope].copy()
    acquired["competence_id"] = acquired.competence_type + ":" + acquired.competence_code
    conflicts = acquired.groupby("competence_id").competence_name.nunique()
    if conflicts.gt(1).any():
        raise DataValidationError("同じ力量タイプ・コードに複数の力量名があります")
    duplicates = int(acquired.duplicated(["member_code", "competence_id"]).sum())
    records = acquired.groupby(["member_code", "competence_id"], as_index=False).value.max()
    competences = acquired[
        ["competence_id", "competence_type", "competence_code", "competence_name"]
    ]
    competences = (
        competences.drop_duplicates("competence_id")
        .sort_values("competence_id")
        .reset_index(drop=True)
    )
    member_index = {code: i for i, code in enumerate(members.member_code)}
    skill_index = {code: i for i, code in enumerate(competences.competence_id)}
    matrix = csr_matrix(
        (
            records.value.astype(float),
            (
                records.member_code.map(member_index).to_numpy(dtype=int),
                records.competence_id.map(skill_index).to_numpy(dtype=int),
            ),
        ),
        shape=(len(members), len(competences)),
    )
    matrix.eliminate_zeros()
    return SkillnoteData(
        members,
        competences,
        matrix,
        {
            "input_members": total_members,
            "active_members": len(members),
            "excluded_members": total_members - len(members),
            "input_acquisitions": len(acquired_frame),
            "excluded_inactive_records": excluded,
            "duplicate_acquisitions": duplicates,
            "unnamed_records": int(unnamed.sum()),
            "competences": len(competences),
            "members_without_acquisitions": int((matrix.getnnz(axis=1) == 0).sum()),
        },
        as_of,
    )


FILENAMES = {
    "members": ("member_skillnote.csv", "member.csv"),
    "acquired": ("acquiredCompetenceLevel.csv", "acquired_competence.csv"),
    "SKILL": ("skill_skillnote.csv", "skill.csv"),
    "EDUCATION": ("education_skillnote.csv", "education.csv"),
    "LICENSE": ("license_skillnote.csv", "license.csv"),
}


def load_directory(
    directory: Path, *, encoding: str = "utf-8-sig", as_of: date | None = None
) -> SkillnoteData:
    frames = {}
    for key, names in FILENAMES.items():
        matches = [directory / name for name in names if (directory / name).is_file()]
        if len(matches) > 1:
            raise DataValidationError(
                f"{key}: 入力が複数あります。1ファイルにしてください: {names}"
            )
        if not matches:
            if key in ("members", "acquired"):
                raise DataValidationError(f"必須CSVがありません: {directory / names[0]}")
            continue
        frames[key] = read_csv(matches[0], encoding)
    return prepare_data(frames.pop("members"), frames.pop("acquired"), frames, as_of=as_of)
