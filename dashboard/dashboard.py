import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import statsmodels.api as sm
import numpy as np

sns.set(style='dark')

month_mapping = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'June',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}

merged_df = pd.read_csv('dashboard/main_data.csv')

def create_daily_rentals_df(df):
    daily_rentals = df.groupby('date')['count_y'].sum().reset_index()
    return daily_rentals

def create_monthly_rentals_df(df):
    df['month'] = pd.to_datetime(df['date']).dt.month
    monthly_rentals = df.groupby('month')['count_y'].sum().reset_index()
    monthly_rentals['month'] = monthly_rentals['month'].map(month_mapping)
    return monthly_rentals

daily_rentals_df = create_daily_rentals_df(merged_df)

min_date = merged_df["date"].min()
max_date = merged_df["date"].max()

with st.sidebar:
    start_date, end_date = st.date_input(
        label='Select Time Span:',
        min_value=pd.to_datetime(min_date),
        max_value=pd.to_datetime(max_date),
        value=(pd.to_datetime(min_date), pd.to_datetime(max_date))
    )

main_df = merged_df[(merged_df["date"] >= str(start_date)) & 
                    (merged_df["date"] <= str(end_date))]

daily_rentals_df = create_daily_rentals_df(main_df)

st.header('Dicoding Collection Dashboard :sparkles:')
st.subheader('Daily Rentals')
col1 = st.columns(1)

with col1[0]:
    total_rentals = daily_rentals_df['count_y'].sum()
    st.metric("Total Rentals", value=total_rentals)

fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(daily_rentals_df['date'], daily_rentals_df['count_y'], marker='o', linewidth=2, color="#90CAF9")
ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)

st.pyplot(fig)

monthly_rentals = create_monthly_rentals_df(merged_df)

st.subheader('Pada bulan apa saja harga sewa sepeda paling tinggi dan paling rendah dalam 2 tahun terakhir?')

option = st.radio("Select view:", ('Best Month :chart_with_upwards_trend:', 'Worst Month :chart_with_downwards_trend:'))

fig, ax = plt.subplots(figsize=(24, 6))

if option == 'Best Month :chart_with_upwards_trend:':
    colors = ["#72BCD4"] + ["#D3D3D3"] * 4
    sns.barplot(x="count_y", y="month", data=monthly_rentals.nlargest(5, 'count_y'), palette=colors, ax=ax)
    ax.set_title("Best Month of Bike Sharing", fontsize=18)
else:
    colors = ["#C7253E"] + ["#D3D3D3"] * 4
    sns.barplot(x="count_y", y="month", data=monthly_rentals.nsmallest(5, 'count_y'), palette=colors, ax=ax)
    ax.set_title("Worst Month of Bike Sharing", fontsize=18)

st.pyplot(fig)

st.subheader('Bagaimana cuaca "temp, atemp, humidity, dan windspeed" dapat mempengaruhi tingkat sewa sepeda?')

option = st.radio("Select view:", ('Temperature :thermometer:', 'Humidity :fog:', 'Wind Speed :tornado:'))

fig, ax = plt.subplots(figsize=(24, 6))

if option == 'Temperature :thermometer:':
    bins = np.arange(main_df['temp_y'].min(), main_df['temp_y'].max() + 1, 1)
    main_df['temp_bin'] = pd.cut(main_df['temp_y'], bins)
    temp_grouped = main_df.groupby('temp_bin')['count_y'].sum().reset_index()
    max_temp_bin = temp_grouped.loc[temp_grouped['count_y'].idxmax()]

    sns.lineplot(data=main_df, x='temp_y', y='count_y', ax=ax, color="#FF5733")
    ax.set_title("Bike Sharing vs Temperature", fontsize=18)

elif option == 'Humidity :fog:':
    bins = np.arange(main_df['humidity_y'].min(), main_df['humidity_y'].max() + 1, 1)
    main_df['humidity_bin'] = pd.cut(main_df['humidity_y'], bins)
    humidity_grouped = main_df.groupby('humidity_bin')['count_y'].sum().reset_index()
    max_humidity_bin = humidity_grouped.loc[humidity_grouped['count_y'].idxmax()]

    sns.lineplot(data=main_df, x='humidity_y', y='count_y', ax=ax, color="#33FF57")
    ax.set_title("Bike Sharing vs Humidity", fontsize=18)

elif option == 'Wind Speed :tornado:':
    if 'windspeed_y' in main_df.columns:
        bins = np.arange(main_df['windspeed_y'].min(), main_df['windspeed_y'].max() + 1, 1)
        main_df['windspeed_bin'] = pd.cut(main_df['windspeed_y'], bins)
        windspeed_grouped = main_df.groupby('windspeed_bin')['count_y'].sum().reset_index()
        max_windspeed_bin = windspeed_grouped.loc[windspeed_grouped['count_y'].idxmax()]

        sns.lineplot(data=main_df, x='windspeed_y', y='count_y', ax=ax, color="#3357FF")
        ax.set_title("Bike Sharing vs Wind Speed", fontsize=18)

st.pyplot(fig)

if option == 'Temperature :thermometer:':
    st.metric("Highest Rentals", f"{max_temp_bin['temp_bin']} with {max_temp_bin['count_y']} rentals.")
    
elif option == 'Humidity :fog:':
    st.metric("Highest Rentals", f"{max_humidity_bin['humidity_bin']} with {max_humidity_bin['count_y']} rentals.")

elif option == 'Wind Speed :tornado:':
    if 'windspeed_y' in main_df.columns:
        st.metric("Highest Rentals", f"{max_windspeed_bin['windspeed_bin']} with {max_windspeed_bin['count_y']} rentals.")
    else:
        st.error("Wind Speed data not available.")

matrix_df = merged_df[['temp_y', 'atemp_y', 'humidity_y', 'windspeed_y', 'count_y']].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(matrix_df, annot=True, cmap='coolwarm', ax=ax)
ax.set_title('Correlation Matrix')
st.pyplot(fig)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

sns.lineplot(data=merged_df, x='temp_y', y='count_y', ax=axs[0])
axs[0].set_title('Bike Sharing vs Temperature')

sns.lineplot(data=merged_df, x='humidity_y', y='count_y', ax=axs[1])
axs[1].set_title('Bike Sharing vs Humidity')

sns.lineplot(data=merged_df, x='windspeed_y', y='count_y', ax=axs[2])
axs[2].set_title('Bike Sharing vs Windspeed')

plt.tight_layout()
st.pyplot(fig)

X = merged_df[['temp_y', 'atemp_y', 'humidity_y', 'windspeed_y']]
y = merged_df['count_y']

X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

st.subheader("Regression Model Summary")
st.text(model.summary().as_text())

st.write("""
- Pada bulan apa saja harga sewa sepeda paling tinggi dan paling rendah dalam 2 tahun terakhir?
  - Melakukan group data terlebih dahulu pada bulan dan count_y yang merupakan totalan dari jam penyewaan.
  - Membuat subplot dalam memberikan visualisasi data dari Best month of bike sharing yaitu, bulan July. Worst month of bike sharing yaitu, bulan January.
  - Mengurutkan bulan yang awalnya diurutkan berdasarkan abjad menjadi January-December.
  - Membuat grafik dalam melihat penyewaan dalam 2 tahun terakhir pada bulan January-December.

- Bagaimana cuaca "temp, atemp, humidity, dan windspeed" dapat mempengaruhi tingkat sewa sepeda?
  - Membuat correlation matrix untuk melihat regresi tingkat penyewaan sepeda yang paling sering muncul dalam hubungan cuaca.
  - Korelasi antar temp_y dengan count_y memiliki korelasi positif satu sama lain, dimana semakin tinggi suhu, tingkat penyewaan sepeda semakin meningkat.
  - Korelasi antara atemp_y dengan count_y memiliki korelasi positif, semakin tinggi atemp maka semakin tinggi jumlah penyewaan sepeda.
  - Korelasi antara humidity_y dengan count_y memiliki korelasi negatif. Apabila tingkat humidity rendah dan tinggi maka, tingkat penyewaan sepeda akan berkurang. Apabila humidity di skala 0.4 sampai 0.6 maka tingkat penyewaan sepeda akan banyak (humidity normal).
  - Korelasi antara windspeed_y dengan count_y memiliki korelasi positif. Tingkat penyewaan sepeda akan tinggi apabila windspeed rendah dan menurun apabila tingkat windspeed meningkat.
""")