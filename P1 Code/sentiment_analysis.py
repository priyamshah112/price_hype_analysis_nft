import np
import pandas as pd
from textblob import TextBlob

files = ['Polkamon_March_April_2021_data', 'Polkamon_May_June_2021_data']
for file in files:
    df = pd.read_csv(f"{file}.csv")
    df["sentiment_score"] = df["Text"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    df["sentiment"] = np.select([df["sentiment_score"] < 0, df["sentiment_score"] == 0, df["sentiment_score"] > 0],
                                ['neg', 'neu', 'pos'])
    print(df)
    df.to_csv(f"{file}_sentiment.csv")
