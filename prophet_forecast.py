import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Prophet Forecast") \
    .getOrCreate()

# Assuming your data is stored in a Parquet file or DataFrame
# Load the data (replace with actual path to your data)
df = spark.read.parquet("/home/ymusic7/metro_restaurants/balanced_reviews.parquet").toPandas()

# Extract year and month from the date column
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

# Calculate the average sentiment per month for each keyword (Japanese/Chinese)
df_monthly_sentiment = df.groupby(['year', 'month', 'keyword']).agg({
    'sentiment': 'mean'
}).reset_index()

# Create a Prophet model for sentiment forecasting
japanese_df = df_monthly_sentiment[df_monthly_sentiment['keyword'] == 'japanese restaurant'][['date', 'sentiment']]
chinese_df = df_monthly_sentiment[df_monthly_sentiment['keyword'] == 'chinese restaurant'][['date', 'sentiment']]

# Prophet requires columns to be renamed as 'ds' for dates and 'y' for the values
japanese_df.rename(columns={'date': 'ds', 'sentiment': 'y'}, inplace=True)
chinese_df.rename(columns={'date': 'ds', 'sentiment': 'y'}, inplace=True)

# Fit Prophet model for both cuisines
prophet_japanese = Prophet(yearly_seasonality=True, daily_seasonality=False)
prophet_japanese.fit(japanese_df)
future_japanese = prophet_japanese.make_future_dataframe(japanese_df, periods=12, freq='M')
forecast_japanese = prophet_japanese.predict(future_japanese)

prophet_chinese = Prophet(yearly_seasonality=True, daily_seasonality=False)
prophet_chinese.fit(chinese_df)
future_chinese = prophet_chinese.make_future_dataframe(chinese_df, periods=12, freq='M')
forecast_chinese = prophet_chinese.predict(future_chinese)

# Plot the results
plt.figure(figsize=(12, 6))

# Japanese Restaurant Forecast
plt.plot(forecast_japanese['ds'], forecast_japanese['yhat'], label="Japanese Forecast", color='blue')

# Chinese Restaurant Forecast
plt.plot(forecast_chinese['ds'], forecast_chinese['yhat'], label="Chinese Forecast", color='green')

plt.title("Sentiment Forecast (Japanese vs Chinese Restaurants)")
plt.xlabel("Date")
plt.ylabel("Average Sentiment Score")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()

# Save the plot as an image file (e.g., PNG)
plt.savefig('/home/ymusic7/forecast_sentiment_plot.png', dpi=300)

# Show the plot
plt.show()

