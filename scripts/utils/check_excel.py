import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_excel('Book1.xlsx')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('\nFirst row:')
for col in df.columns:
    print(f'\nColumn: {col[:50]}')
    print(f'Value: {str(df[col].iloc[0])[:200]}...')

