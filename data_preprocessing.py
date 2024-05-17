import pandas as pd
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def preprocess_data(categorical_features, numerical_features):
    conn = sqlite3.connect('/mnt/data/twitch_data.db')
    query = '''
    SELECT
        Channels.Channel,
        Channels.Language,
        Channels.Partnered,
        Channels.Mature,
        ChannelStats.WatchTimeMinutes,
        ChannelStats.StreamTimeMinutes,
        ChannelStats.PeakViewers,
        ChannelStats.AverageViewers,
        ChannelStats.Followers,
        ChannelStats.FollowersGained,
        ChannelStats.ViewsGained
    FROM
        Channels
    JOIN
        ChannelStats
    ON
        Channels.id = ChannelStats.channel_id
    '''
    df = pd.read_sql_query(query, conn)
    conn.close()

    df['Partnered'] = df['Partnered'].astype('category')
    df['Mature'] = df['Mature'].astype('category')

    X = df.drop(columns=['AverageViewers'])
    y = df['AverageViewers']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=df['Partnered'], random_state=42)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor
