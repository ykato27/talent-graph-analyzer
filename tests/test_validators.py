"""
validators.py のテスト
"""

import pytest
import numpy as np
from causal_inference import validators
from causal_inference.exceptions import InvalidInputError, InsufficientDataError


class TestValidateArrayLengths:
    """validate_array_lengths のテスト"""

    def test_same_length(self):
        """同じ長さの配列は成功"""
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        validators.validate_array_lengths(a, b)  # エラーが発生しないことを確認

    def test_different_length(self):
        """異なる長さの配列は失敗"""
        a = np.array([1, 2, 3])
        b = np.array([4, 5])

        with pytest.raises(InvalidInputError, match="Array length mismatch"):
            validators.validate_array_lengths(a, b)

    def test_with_names(self):
        """名前付きエラーメッセージ"""
        a = np.array([1, 2, 3])
        b = np.array([4, 5])

        with pytest.raises(InvalidInputError, match="y=3, X=2"):
            validators.validate_array_lengths(a, b, names=["y", "X"])

    def test_empty_input(self):
        """空の入力は成功"""
        validators.validate_array_lengths()  # エラーが発生しないことを確認


class TestValidatePositiveInteger:
    """validate_positive_integer のテスト"""

    def test_valid_positive(self):
        """正の整数は成功"""
        validators.validate_positive_integer(5, "n_samples")

    def test_zero(self):
        """ゼロは失敗（デフォルトmin_value=1）"""
        with pytest.raises(InvalidInputError):
            validators.validate_positive_integer(0, "n_samples")

    def test_negative(self):
        """負の値は失敗"""
        with pytest.raises(InvalidInputError):
            validators.validate_positive_integer(-1, "n_samples")

    def test_custom_min_value(self):
        """カスタム最小値"""
        validators.validate_positive_integer(10, "n_samples", min_value=5)

        with pytest.raises(InvalidInputError):
            validators.validate_positive_integer(3, "n_samples", min_value=5)

    def test_non_integer(self):
        """整数でない値は失敗"""
        with pytest.raises(InvalidInputError, match="must be an integer"):
            validators.validate_positive_integer(5.5, "n_samples")


class TestValidateProbability:
    """validate_probability のテスト"""

    def test_valid_probability(self):
        """有効な確率値は成功"""
        validators.validate_probability(0.5, "alpha")
        validators.validate_probability(0.0, "alpha")
        validators.validate_probability(1.0, "alpha")

    def test_out_of_range(self):
        """範囲外の値は失敗"""
        with pytest.raises(InvalidInputError, match="must be in"):
            validators.validate_probability(1.5, "alpha")

        with pytest.raises(InvalidInputError):
            validators.validate_probability(-0.1, "alpha")


class TestValidateArrayNoNaN:
    """validate_array_no_nan のテスト"""

    def test_no_nan(self):
        """NaNなしの配列は成功"""
        arr = np.array([1.0, 2.0, 3.0])
        validators.validate_array_no_nan(arr, "data")

    def test_with_nan(self):
        """NaN含む配列は失敗"""
        arr = np.array([1.0, np.nan, 3.0])

        with pytest.raises(InvalidInputError, match="contains.*NaN"):
            validators.validate_array_no_nan(arr, "data")


class TestValidateArrayNoInf:
    """validate_array_no_inf のテスト"""

    def test_no_inf(self):
        """無限大なしの配列は成功"""
        arr = np.array([1.0, 2.0, 3.0])
        validators.validate_array_no_inf(arr, "data")

    def test_with_inf(self):
        """無限大含む配列は失敗"""
        arr = np.array([1.0, np.inf, 3.0])

        with pytest.raises(InvalidInputError, match="contains.*infinite"):
            validators.validate_array_no_inf(arr, "data")


class TestValidateSufficientData:
    """validate_sufficient_data のテスト"""

    def test_sufficient(self):
        """十分なデータは成功"""
        validators.validate_sufficient_data(100, 50)

    def test_insufficient(self):
        """不十分なデータは失敗"""
        with pytest.raises(InsufficientDataError, match="Insufficient data"):
            validators.validate_sufficient_data(30, 50)


class TestValidateBinaryArray:
    """validate_binary_array のテスト"""

    def test_valid_binary(self):
        """有効な二値配列は成功"""
        arr = np.array([0, 1, 1, 0, 1])
        validators.validate_binary_array(arr, "treatment")

    def test_invalid_values(self):
        """0と1以外の値は失敗"""
        arr = np.array([0, 1, 2, 0])

        with pytest.raises(InvalidInputError, match="must be binary"):
            validators.validate_binary_array(arr, "treatment")


class TestValidateGammaValues:
    """validate_gamma_values のテスト"""

    def test_valid_gamma(self):
        """有効なGamma値は成功"""
        validators.validate_gamma_values([1.0, 1.5, 2.0])

    def test_empty_list(self):
        """空のリストは失敗"""
        with pytest.raises(InvalidInputError, match="must not be empty"):
            validators.validate_gamma_values([])

    def test_gamma_less_than_one(self):
        """1未満のGamma値は失敗"""
        with pytest.raises(InvalidInputError, match="must be >= 1.0"):
            validators.validate_gamma_values([0.5, 1.5])


class TestValidate2DArray:
    """validate_2d_array のテスト"""

    def test_valid_2d(self):
        """有効な2次元配列は成功"""
        arr = np.array([[1, 2], [3, 4]])
        validators.validate_2d_array(arr, "X")

    def test_1d_array(self):
        """1次元配列は失敗"""
        arr = np.array([1, 2, 3])

        with pytest.raises(InvalidInputError, match="must be a 2D array"):
            validators.validate_2d_array(arr, "X")

    def test_3d_array(self):
        """3次元配列は失敗"""
        arr = np.array([[[1, 2]], [[3, 4]]])

        with pytest.raises(InvalidInputError, match="must be a 2D array"):
            validators.validate_2d_array(arr, "X")


class TestValidateClusters:
    """validate_clusters のテスト"""

    def test_sufficient_clusters(self):
        """十分なクラスター数は成功"""
        clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        validators.validate_clusters(clusters, min_clusters=3)

    def test_insufficient_clusters(self):
        """不十分なクラスター数は失敗"""
        clusters = np.array([0, 0, 1, 1])

        with pytest.raises(InvalidInputError, match="Insufficient number of clusters"):
            validators.validate_clusters(clusters, min_clusters=3)
