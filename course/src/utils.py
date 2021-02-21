import pandas as pd
import numpy as np

def prefilter_items(data):
    # Уберем top 3 самых популярных товаров
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    top = popularity.sort_values('quantity', ascending=False).head(3).item_id.tolist()
    data.loc[data['item_id'].isin(top), 'item_id'] = 999999

    # Уберем товары, с общим числом продаж < 50
    less = popularity[popularity.quantity < 50].item_id.tolist()
    data.loc[data['item_id'].isin(less), 'item_id'] = 999999

    # Уберем товары, которые не продавались за последние 12 месяцев
    actuality = data.groupby('item_id')['day'].nunique().reset_index()
    top_actual = actuality[actuality.day > 365].item_id.tolist()
    data.loc[data['item_id'].isin(top_actual), 'item_id'] = 999999

    # Уберем товары, которые стоят < 1$
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    low_price = data[data['price'] < 1].item_id.tolist()
    data.loc[data['item_id'].isin(low_price), 'item_id'] = 999999

    return data

def postfilter_items(user_id, recommednations):
    pass