import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

st.title('End-to-End Sales Forecasting & Demand Intelligence System')
st.sidebar.title('Navigation')
page = st.sidebar.radio(
    "",    
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

#Loading dataset and parsing dated columns
df = pd.read_csv(
    'train.csv',
    parse_dates=['Order Date', 'Ship Date'],
    date_format='%d/%m/%Y'
)

#Data Cleaning

df['Postal Code'] = df['Postal Code'].astype('string')
df['Postal Code'] = df['Postal Code'].str.replace('.0', '', regex=False)

#Drop unnescessary columns
dropcols = [
    'Row ID',
    'Order ID',
    'Customer ID',
    'Product ID',
    'Product Name',
    'Customer Name',
    'Country'
]

df.drop(dropcols, axis=1, inplace=True)

#Adding Date Features
df['OrderDate_Year'] = df['Order Date'].dt.year
df['OrderDate_Month'] = df['Order Date'].dt.month
df['OrderDate_Quarter'] = df['Order Date'].dt.quarter
df['OrderDate_WeekNo'] = df['Order Date'].dt.isocalendar().week
df['OrderDate_DayOfWeek'] = df['Order Date'].dt.dayofweek

df['ShipDate_Year'] = df['Ship Date'].dt.year
df['ShipDate_Month'] = df['Ship Date'].dt.month
df['ShipDate_Quarter'] = df['Ship Date'].dt.quarter
df['ShipDate_WeekNo'] = df['Ship Date'].dt.isocalendar().week
df['ShipDate_DayOfWeek'] = df['Ship Date'].dt.dayofweek

#Function for getting season by passing month
def extract_season(monthlist):
    season = []
    for mon in monthlist:
        if mon in [12, 1, 2]:
            season.append("Winter")
        elif mon in [3, 4, 5]:
            season.append("Spring")
        elif mon in [6, 7, 8]:
            season.append("Summer")
        else:
            season.append("Autumn")
    return season

df['OrderDate_Season'] = extract_season(df['OrderDate_Month'])
df['ShipDate_Season'] = extract_season(df['ShipDate_Month'])

#Replacing Null values by actual Postal Code
df.loc[df['Postal Code'].isnull(), 'Postal Code'] = "05401"

#Dropping duplicates
df = df.drop_duplicates()

#Setting Order Date as index
df.set_index('Order Date', inplace=True)

if page == 'Sales Overview':
    st.markdown("## Sales Overview Dashboard")

    #Yearly Sales Bar Chart
    st.markdown("### *Total Sales by Year*")
    yearly_sales = df.groupby('OrderDate_Year')['Sales'].sum()
    st.bar_chart(yearly_sales)

    # Monthly Sales Line Chart
    st.divider()
    st.markdown("### *Monthly Sales*")
    monthly_sales = df['Sales'].resample('MS').sum()
    st.line_chart(monthly_sales)

    #Sales by Region & category Chart
    st.divider()
    st.markdown("### *Sales by Region and Category*")
    region = st.selectbox("Select Region", ["All"]+list(df['Region'].unique()))
    category = st.selectbox("Select Category",["All"]+list(df['Category'].unique()))

    filtered_df = df.copy()
    if region != "All":
        filtered_df = filtered_df[filtered_df['Region']==region]
    if category != "All":
        filtered_df = filtered_df[filtered_df['Category']==category]
    
    if region!="All" and category=="All":
        sales = filtered_df.groupby('Category')['Sales'].sum()
        st.bar_chart(sales)
    elif region=="All" and category!="All":
        sales = filtered_df.groupby('Region')['Sales'].sum()
        st.bar_chart(sales)
    elif region=="All" and category=="All":
        pass
    else:
        st.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
        st.metric("Average Order Values", f"${filtered_df['Sales'].mean():,.2f}")
        st.metric("Number of Orders", f"{len(filtered_df)}")
elif page == 'Forecast Explorer':
    st.markdown("## Forecast Explorer")
    st.markdown('### *Using Prophet Model*')
    
    #Select Box for selecting Category or Region
    cat_reg = st.selectbox("Category/Region", ["Category","Region"])
    new_df = df.copy()

    #Select box for selecting specific category/region
    if cat_reg == "Region":
        region = st.selectbox("Select Region", list(df['Region'].unique()))
        st.metric("Region", f"{region}")
        new_df = df[df['Region']==region]['Sales'].resample('MS').sum().to_frame()  #Updating dataframe  
    elif cat_reg == "Category":
        category = st.selectbox("Select Category", list(df['Category'].unique()))
        st.metric("Category", f"{category}")
        new_df = df[df['Category']==category]['Sales'].resample('MS').sum().to_frame()  #Updating dataframe


    #Training Prophet Model
    new_df.reset_index(inplace=True)
    new_df.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
    model.fit(new_df)   

    #Forecast Months Slider
    forecast_period = st.slider("Select Forecast period (Months)",min_value=1,max_value=3,value=1,step=1)

    #Generating future dates
    future = model.make_future_dataframe(periods=forecast_period, freq='MS')
    forecast = model.predict(future)    #Predicting Sales

    forecast = forecast.tail(forecast_period).copy()
    forecast["ds"] = forecast["ds"].dt.strftime("%m/%Y")
    forecast.rename(columns={"yhat": "Forecasted Sales"}, inplace=True)
    forecast.rename(columns={"ds": "Month"}, inplace=True)
    forecast = forecast.set_index("Month")["Forecasted Sales"]

    #Plotting bar/line chart
    if forecast_period==1:
        st.bar_chart(forecast)
    else:
        st.line_chart(forecast)

    #Metrics- we'll split data into train-test(80/20) and retrain to get metrics
    st.divider()

    #Train-Test Split
    split = int(len(new_df) * 0.8)
    train = new_df.iloc[:split]
    test = new_df.iloc[split:]

    #Training Model on train data
    model_2 = Prophet(yearly_seasonality=True,weekly_seasonality=False,daily_seasonality=False)
    model_2.fit(train)

    predicted = model_2.predict(test[["ds"]])   #Predicting

    #MAE and RMSE Calculation
    mae = mean_absolute_error(test["y"], predicted["yhat"])
    rmse = np.sqrt(mean_squared_error(test["y"], predicted["yhat"]))

    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"{mae:,.2f}")
    col2.metric("Root Mean Squared Error", f"{rmse:,.2f}")
elif page=="Anomaly Report":
    st.markdown("## Anomaly Report")

    #Using Isolation Forest to detect anomalies in weekly sales data
    weekly_sales = df['Sales'].resample('W').sum().to_frame()  #Weekly Totals
    iso_model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)    #Very few, almost 5% of the data can be outliers/anomalies. Therefore, contamination=0.05
    iso_model.fit(weekly_sales)
    anomalies = iso_model.predict(weekly_sales)     #Predict Anomalies
    anomalies_df = pd.DataFrame({'Sales':weekly_sales['Sales'], 'Anomaly': anomalies})    
    anomalies_df["Anomaly"] = anomalies_df["Anomaly"].map({-1: "Anomaly",1: "Normal"})

    #Plotting Figure
    st.markdown("### *Anomalies of Weekly Sales Data*")
    fig = px.scatter(
        anomalies_df,x=anomalies_df.index,y="Sales",color="Anomaly",
        color_discrete_map={"Anomaly": "red","Normal": "blue"},
        labels={
            "x": "Week",
            "Sales": "Sales"
        }
    )
    fig.update_layout(xaxis_title="Week",yaxis_title="Sales")
    st.plotly_chart(fig, use_container_width=True)
    
    #Anomalies Table
    st.divider()
    st.markdown("### *Detected Anomalies Details*")
    anomalies_table = anomalies_df[anomalies_df['Anomaly']=="Anomaly"].copy()    
    anomalies_table['OrderDate_WeekNumber'] = anomalies_table.index.isocalendar().week
    anomalies_table.index = anomalies_table.index.strftime('%Y-%m-%d')
    anomalies_table = anomalies_table.drop(columns="Anomaly")
    st.write(anomalies_table)
else:
    st.markdown("## Product Demand Segments")
    #Total Sales
    subcategory = df.groupby('Sub-Category')['Sales'].sum().to_frame()
    subcategory.reset_index(inplace=True)
    subcategory.columns = ['Sub-Category', 'Total Sales']   #Renaming column

    #Growth
    growth = df.groupby(['Sub-Category','OrderDate_Year'])['Sales'].sum().to_frame()
    growth.reset_index(inplace=True)
    growth['Growth_Rate'] = growth.groupby('Sub-Category')['Sales'].pct_change()
    growth_rate = growth.groupby('Sub-Category')['Growth_Rate'].mean()
    subcategory['Growth Rate'] = subcategory['Sub-Category'].map(growth_rate)

    #Volatility
    volatility = df.groupby(['Sub-Category','OrderDate_Year','OrderDate_Month'])['Sales'].sum().to_frame()
    volatility.reset_index(inplace=True)
    sales_volatility = volatility.groupby('Sub-Category')['Sales'].std()    #Calculating volatility using std() - standard deviation
    subcategory['Sales Volatility'] = subcategory['Sub-Category'].map(sales_volatility)

    #Average Sales
    subcategory['Average Sales'] = subcategory['Sub-Category'].map(df.groupby('Sub-Category')['Sales'].mean())

    #Scaling the features
    X = subcategory.drop(columns='Sub-Category')
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #Applying K-Means
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)
    predicted_Cluster = kmeans.predict(X_scaled)

    subcategory['Cluster'] = predicted_Cluster
    map_Cluster = {
        0: 'High Volume, High Volatility',
        1: 'Low Volume, Stable Demand',
        2: 'High Volume, Stable Demand',
        3: 'Growing Demand'
    }
    subcategory['Cluster'] = subcategory['Cluster'].map(map_Cluster)

    #Applying PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    #Plotting graph
    st.markdown("### *K-Means Cluster of Product Sub-Categories*")
    plot_df = subcategory.copy()
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        text="Sub-Category",
    )

    fig.update_traces(
        textposition="top center",
        textfont_size=9
    )

    fig.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### *Cluster Details of Sub-Categories*")
    st.write(subcategory[['Sub-Category','Cluster']])
