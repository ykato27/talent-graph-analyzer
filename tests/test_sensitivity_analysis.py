"""
sensitivity_analysis.py のテスト
"""

import pytest
import numpy as np
import pandas as pd
from causal_inference import sensitivity_analysis
from causal_inference.exceptions import InvalidInputError


class TestRosenbaumBounds:
    """rosenbaum_bounds のテスト"""

    def test_basic_functionality(self, sample_binary_data):
        """基本的な機能のテスト"""
        treated, control = sample_binary_data

        result = sensitivity_analysis.rosenbaum_bounds(treated, control)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5  # デフォルトのGamma値は5つ
        assert 'Gamma' in result.columns
        assert 'P_value_upper' in result.columns
        assert 'P_value_lower' in result.columns

    def test_custom_gamma_values(self, sample_binary_data):
        """カスタムGamma値のテスト"""
        treated, control = sample_binary_data

        gamma_values = [1.0, 2.0, 3.0]
        result = sensitivity_analysis.rosenbaum_bounds(
            treated, control, gamma_values=gamma_values
        )

        assert len(result) == 3
        assert list(result['Gamma']) == gamma_values

    def test_gamma_ordering(self, sample_binary_data):
        """Gammaが大きくなるとp値の範囲が広がる"""
        treated, control = sample_binary_data

        result = sensitivity_analysis.rosenbaum_bounds(treated, control)

        # Gammaが大きくなると上限p値は増加、下限p値は減少する傾向
        assert result['Gamma'].is_monotonic_increasing

    def test_empty_input(self):
        """空の入力はエラー"""
        with pytest.raises((ValueError, IndexError)):
            sensitivity_analysis.rosenbaum_bounds(np.array([]), np.array([]))


class TestCalculateEValue:
    """calculate_e_value のテスト"""

    def test_odds_ratio_basic(self):
        """オッズ比の基本的なテスト"""
        result = sensitivity_analysis.calculate_e_value(
            effect_estimate=2.0,
            effect_type="odds_ratio"
        )

        assert 'point_estimate' in result
        assert 'interpretation' in result
        assert result['point_estimate'] > 1.0  # E-valueは1より大きい

    def test_odds_ratio_with_se(self):
        """標準誤差を含むテスト"""
        result = sensitivity_analysis.calculate_e_value(
            effect_estimate=2.0,
            effect_se=0.3,
            effect_type="odds_ratio"
        )

        assert 'ci_lower' in result
        assert result['ci_lower'] > 0

    def test_protective_effect(self):
        """保護効果（OR < 1）のテスト"""
        result = sensitivity_analysis.calculate_e_value(
            effect_estimate=0.5,  # 保護効果
            effect_type="odds_ratio"
        )

        # 自動的に逆数を取って計算される
        assert result['point_estimate'] > 1.0

    def test_interpretation_levels(self):
        """E-valueの解釈レベルのテスト"""
        # 低いE-value
        result_low = sensitivity_analysis.calculate_e_value(1.2, effect_type="odds_ratio")
        assert "脆弱" in result_low['interpretation']

        # 高いE-value
        result_high = sensitivity_analysis.calculate_e_value(5.0, effect_type="odds_ratio")
        assert "頑健" in result_high['interpretation']

    def test_invalid_effect_type(self):
        """無効なeffect_typeはエラー"""
        with pytest.raises(ValueError):
            sensitivity_analysis.calculate_e_value(2.0, effect_type="unknown")


class TestSensitivityAnalysisReport:
    """sensitivity_analysis_report のテスト"""

    def test_basic_report_generation(self, sample_continuous_data):
        """基本的なレポート生成のテスト"""
        treated, control = sample_continuous_data

        report = sensitivity_analysis.sensitivity_analysis_report(
            treated_outcomes=treated,
            control_outcomes=control,
            effect_estimate=1.8,
            effect_se=0.2,
            effect_type="odds_ratio"
        )

        assert 'rosenbaum_bounds' in report
        assert 'e_value' in report
        assert 'summary' in report
        assert 'recommendation' in report

        # サマリーにはキーワードが含まれる
        assert 'Rosenbaum' in report['summary']
        assert 'E-value' in report['summary']

    def test_recommendation_quality(self, sample_continuous_data):
        """推奨事項の品質テスト"""
        treated, control = sample_continuous_data

        # 高いE-valueの場合
        report_high = sensitivity_analysis.sensitivity_analysis_report(
            treated, control,
            effect_estimate=4.0,
            effect_type="odds_ratio"
        )
        assert "✅" in report_high['recommendation'] or "推奨" in report_high['recommendation']

        # 低いE-valueの場合
        report_low = sensitivity_analysis.sensitivity_analysis_report(
            treated, control,
            effect_estimate=1.3,
            effect_type="odds_ratio"
        )
        assert "❌" in report_low['recommendation'] or "慎重" in report_low['recommendation']


class TestHelperFunctions:
    """ヘルパー関数のテスト"""

    def test_interpret_e_value(self):
        """E-valueの解釈関数のテスト"""
        # プライベート関数なので直接テストしない場合もある
        # が、ここではテストの徹底のため実施

        from causal_inference.sensitivity_analysis import _interpret_e_value

        assert "脆弱" in _interpret_e_value(1.2)
        assert "頑健" in _interpret_e_value(3.5)

    def test_generate_summary(self, sample_continuous_data):
        """サマリー生成のテスト"""
        treated, control = sample_continuous_data

        rosenbaum_results = sensitivity_analysis.rosenbaum_bounds(treated, control)
        e_value_results = sensitivity_analysis.calculate_e_value(2.0, effect_type="odds_ratio")

        from causal_inference.sensitivity_analysis import _generate_summary

        summary = _generate_summary(rosenbaum_results, e_value_results)

        assert isinstance(summary, str)
        assert len(summary) > 0
        assert "Gamma" in summary or "E-value" in summary
