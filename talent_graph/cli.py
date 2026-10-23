"""ネットワークを使用しないローカルコマンド。"""

import argparse
import hashlib
import json
import platform
import sys
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .analysis import analyze
from .data import FILENAMES, DataValidationError, load_directory, normalize_columns, read_csv


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def _fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"file": path.name, "sha256": digest.hexdigest()}


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("1以上を指定してください")
    return result


def _new_output(path):
    if path.exists() or path.is_symlink():
        raise DataValidationError(
            f"出力先は既に存在します。新しいフォルダを指定してください: {path}"
        )


def _references(args):
    if args.reference_file:
        frame = normalize_columns(read_csv(args.reference_file, args.encoding), ("member_code",))
        return frame.member_code.tolist()
    return args.reference


def _input_files(directory):
    return sorted(
        path
        for names in FILENAMES.values()
        for name in names
        if (path := directory / name).is_file()
    )


def _report(data, result, kind):
    metrics = result.metrics["models"]
    lines = [
        "# Skillnote 力量比較結果",
        "",
        f"基準日: {data.as_of.isoformat()} / データ: {'架空のデモ' if kind == 'synthetic' else '利用者指定CSV'}",
        "",
        f"在籍社員: {len(data.members)}名 / 力量: {len(data.competences)}件 / 基準群: {int(result.members.is_reference.sum())}名",
        "",
        "## 結果の読み方",
        "",
        "- member_scores.csv: 指定基準群の平均埋め込みとのコサイン類似度（-1〜1）です。優秀さの確率ではありません。",
        "- 基準群のスコアには本人も含まれます。精度の評価には別の交差検証を使用しています。",
        "- skill_importance.csv: 全力量の保有率差、Wilson信頼区間、Fisher検定、全力量を対象にしたFDR補正です。",
        "- 習得記録がないセルを0として扱います。未評価と未習得は区別できません。",
        "- 教育・資格は習得CSVの行の存在を保有=1として扱います。有効期限はこの入力形式では判定しません。",
        "- 習得記録のない社員の類似度は空欄です。",
        "- 観察データの関連を示す結果です。因果効果や個人の能力を確定する用途には使えません。",
        "",
        "## 検証",
        "",
        "学習には基準群ラベルを使用しません。全社員の力量からグラフを作るtransductive方式です。",
        "基準群の平均とロジスティック回帰は、各分割の訓練ラベルだけを使用します。",
        "将来の新入社員・別企業への精度を測る評価ではありません。",
        "",
    ]
    for name, metric in metrics.items():
        lines.append(
            f"- {name}: AUC={metric['roc_auc']:.3f}, AP={metric['average_precision']:.3f}, "
            f"{metric['samples']}名 / {metric['folds']}分割"
        )
    if not metrics:
        lines.append(f"- 評価不可: {result.metrics['reason']}")
    if result.training["status"] != "trained":
        lines.extend(
            [
                "",
                "比較可能なエッジ・非エッジが足りず、重み学習を実行していません。",
                "類似度は固定乱数の未学習埋め込みによる参考値です。",
            ]
        )
    lines.extend(["", "## 入力品質", "", *[f"- {k}: {v}" for k, v in data.quality.items()], ""])
    return "\n".join(lines)


def _run_analysis(args):
    _new_output(args.output)
    output = args.output.resolve()
    directory = args.data_dir.resolve()
    if output == directory or output.is_relative_to(directory):
        raise DataValidationError("出力先は入力フォルダの外にしてください")
    input_files = _input_files(directory)
    before = [_fingerprint(p) for p in input_files]
    data = load_directory(directory, encoding=args.encoding, as_of=args.as_of)
    reference_before = _fingerprint(args.reference_file) if args.reference_file else None
    selected = _references(args)
    result = analyze(
        data,
        selected,
        epochs=args.epochs,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        neighbors=args.neighbors,
    )
    if before != [_fingerprint(p) for p in input_files]:
        raise DataValidationError("処理中に入力CSVが変更されました。もう一度実行してください")
    if reference_before and reference_before != _fingerprint(args.reference_file):
        raise DataValidationError("処理中に基準群CSVが変更されました")
    kind = "user_provided"
    marker = directory / "synthetic.json"
    if marker.is_file():
        try:
            if json.loads(marker.read_text(encoding="utf-8")).get("inputs") == before:
                kind = "synthetic"
        except ValueError, OSError:
            pass
    metadata = {
        "version": __version__,
        "python": platform.python_version(),
        "created_at": datetime.now(UTC).isoformat(),
        "as_of": data.as_of.isoformat(),
        "data_kind": kind,
        "inputs": before,
        "reference_file": reference_before,
        "reference_codes": selected,
        "parameters": {
            "encoding": args.encoding,
            "epochs": args.epochs,
            "seed": args.seed,
            "hidden_dim": args.hidden_dim,
            "neighbors": args.neighbors,
        },
        "dependencies": {
            name: version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "statsmodels")
        },
        "quality": data.quality,
        "evaluation": result.metrics,
        "training": result.training,
        "embedding_member_codes": data.members.member_code.tolist(),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    result.members.to_csv(output / "member_scores.csv", index=False, encoding="utf-8-sig")
    result.skills.to_csv(output / "skill_importance.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output / "embeddings.npz",
        embeddings=result.embeddings,
        member_codes=np.asarray(data.members.member_code, dtype=str),
    )
    (output / "run.json").write_text(_json(metadata) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(data, result, kind), encoding="utf-8")
    print(f"分析結果を保存しました: {output}")


def _demo(output):
    _new_output(output)
    rng = np.random.default_rng(42)
    codes = [f"{i:04}" for i in range(1, 51)]
    members = pd.DataFrame(
        {
            "メンバーコード  ###[member_code]###": codes,
            "メンバー名  ###[name]###": [f"架空社員{i:02}" for i in range(1, 51)],
            "退職年月日  ###[retired_day]###": [""] * 48 + ["2020-01-01"] * 2,
            "入社年月日  ###[enter_day]###": ["2020-04-01"] * 50,
        }
    )
    rows = []
    for i, member in enumerate(codes):
        for j in range(16):
            kind = "SKILL" if j < 12 else ("EDUCATION" if j < 14 else "LICENSE")
            probability = 0.85 if i < 6 and j < 4 else 0.35
            if rng.random() < probability:
                rows.append(
                    {
                        "メンバーコード": member,
                        "力量タイプ  ###[competence_type]###": kind,
                        "力量コード": f"{j % 12 + 1:03}",
                        "力量名": f"架空{kind}{j + 1:02}",
                        "レベル": str(rng.integers(1, 5)) if kind == "SKILL" else "",
                    }
                )
    output.mkdir(parents=True, exist_ok=False)
    members.to_csv(output / "member_skillnote.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rows).to_csv(
        output / "acquiredCompetenceLevel.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame({"member_code": codes[:6]}).to_csv(
        output / "reference.csv", index=False, encoding="utf-8-sig"
    )
    (output / "synthetic.json").write_text(
        _json({"inputs": [_fingerprint(p) for p in _input_files(output)]}), encoding="utf-8"
    )
    print(f"架空のSkillnote形式CSVを保存しました: {output.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Skillnote CSVをローカルで検証・分析します")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="架空CSVを生成")
    demo.add_argument("--output", type=Path, required=True)
    for command in ("validate", "analyze"):
        cmd = sub.add_parser(
            command, help="CSVの検証" if command == "validate" else "力量比較・グラフ分析"
        )
        cmd.add_argument("--data-dir", type=Path, required=True)
        cmd.add_argument("--encoding", default="utf-8-sig", help="Shift-JISの場合はcp932")
        cmd.add_argument(
            "--as-of", type=date.fromisoformat, default=date.today(), help="基準日 YYYY-MM-DD"
        )
        if command == "analyze":
            group = cmd.add_mutually_exclusive_group(required=True)
            group.add_argument(
                "--reference", nargs="+", help="基準群のメンバーコード（先頭ゼロを保持）"
            )
            group.add_argument("--reference-file", type=Path, help="メンバーコード列を持つCSV")
            cmd.add_argument("--output", type=Path, required=True)
            cmd.add_argument("--epochs", type=_positive, default=100)
            cmd.add_argument("--hidden-dim", type=_positive, default=32)
            cmd.add_argument("--neighbors", type=_positive, default=10)
            cmd.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            _demo(args.output)
        elif args.command == "validate":
            print(
                _json(
                    load_directory(args.data_dir, encoding=args.encoding, as_of=args.as_of).quality
                )
            )
        else:
            _run_analysis(args)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    return 0
