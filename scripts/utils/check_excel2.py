import pandas as pd

df = pd.read_excel('Book1.xlsx')
print('Total rows:', len(df))
print('\nChecking first 5 rows:')
for i in range(min(5, len(df))):
    print(f'\nRow {i}:')
    col0 = str(df.iloc[i, 0])
    col1 = str(df.iloc[i, 1])
    print(f'  Col 0 length: {len(col0)}')
    print(f'  Col 1 length: {len(col1)}')
    print(f'  Col 1 value: {col1[:200]}')
    print(f'  Col 1 full: {repr(col1)}')

