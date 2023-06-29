import pandas as pd

from src.data.utils import get_annotations_ends


def test_get_annotations_ends_no_annotation():
    rows, cols = 5, 1
    data = [0, 0, 0, 0, 0]
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    df = pd.DataFrame(data, columns=['Obstructive'], index=tidx)

    result = get_annotations_ends(df)

    assert result.empty


def test_get_annotations_ends_one_annotation():
    rows, cols = 5, 1
    data = [0, 1, 1, 1, 0]
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    df = pd.DataFrame(data, columns=['Obstructive'], index=tidx)
    print(df)
    result = get_annotations_ends(df)
    print(result)
    assert len(result) == 1
    assert result.loc['2019-01-01 00:00:03', 'Obstructive'] == 1


def test_get_annotations_ends_two_annotation():
    rows, cols = 8, 1
    data = [0, 1, 1, 1, 0, 1, 1, 0]
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    df = pd.DataFrame(data, columns=['Obstructive'], index=tidx)
    print(df)
    result = get_annotations_ends(df)
    print(result)
    assert len(result) == 2
    assert result.loc['2019-01-01 00:00:03', 'Obstructive'] == 1
    assert result.loc['2019-01-01 00:00:06', 'Obstructive'] == 1


def test_get_annotations_ends_annotation_at_end():
    rows, cols = 7, 1
    data = [0, 1, 1, 1, 0, 1, 1]
    tidx = pd.date_range('2019-01-01', periods=rows, freq='S')
    df = pd.DataFrame(data, columns=['Obstructive'], index=tidx)

    result = get_annotations_ends(df)

    assert len(result) == 2
    assert result.loc['2019-01-01 00:00:03', 'Obstructive'] == 1
    assert result.loc['2019-01-01 00:00:06', 'Obstructive'] == 1
