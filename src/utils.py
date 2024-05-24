import pandas as pd


def adjust_column_names(df: pd.DataFrame) -> pd.DataFrame:
    '''
    adjusts the column names.
    '''
    df.columns = df\
        .columns\
        .str\
        .strip()\
        .str\
        .replace(' ', '_')\
        .str\
        .lower()
    
    df.rename(columns={'price_($)': 'price_usd'}, inplace=True)
    
    return df