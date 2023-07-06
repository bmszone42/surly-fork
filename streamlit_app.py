import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

# Sidebar inputs
num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1000, max_value=100000, value=10000)

# User input for cost ranges
overhead_range = st.sidebar.slider('Overhead Range ($)', min_value=2000, max_value=200000, value=(2000, 200000))
cots_chips_range = st.sidebar.slider('COTS Chips Range ($)', min_value=1000, max_value=10000, value=(1000, 10000))
custom_chips_range = st.sidebar.slider('Custom Chips Range ($)', min_value=1000, max_value=10000, value=(1000, 10000))
custom_chips_nre_range = st.sidebar.slider('Custom Chips NRE Range ($)', min_value=1000000, max_value=10000000, value=(1000000, 10000000))
custom_chips_licensing_range = st.sidebar.slider('Custom Chips Licensing Range ($)', min_value=0, max_value=1000000, value=(0, 1000000))
ebrick_chiplets_range = st.sidebar.slider('eBrick Chiplets Range ($)', min_value=20, max_value=150, value=(20, 150))
ebrick_chiplets_licensing_range = st.sidebar.slider('eBrick Chiplets Licensing Range ($)', min_value=0, max_value=1000000, value=(0, 1000000))
osat_range = st.sidebar.slider('OSAT Range ($)', min_value=500000, max_value=750000, value=(500000, 750000))
vv_tests_range = st.sidebar.slider('V&V Tests Range ($)', min_value=500000, max_value=750000, value=(500000, 750000))
profit_margin_range = st.sidebar.slider('Profit Margin Range (%)', min_value=20, max_value=30, value=(20, 30))

# Perform the simulations
@st.cache
def simulate(num_simulations, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range):
    df = pd.DataFrame()
    for _ in range(num_simulations):
        overhead = round(np.random.uniform(*overhead_range), -2)
        cots_chips = round(np.random.randint(1, 6) * np.random.uniform(*cots_chips_range), -2)
        custom_chips = round(np.random.randint(0, 3) * np.random.uniform(*custom_chips_range), -2)
        custom_chips_nre = round(np.random.uniform(*custom_chips_nre_range), -2)
        custom_chips_licensing = round(np.random.uniform(*custom_chips_licensing_range), -2)
        ebrick_chiplets = round(np.random.choice(np.arange(16, 257, 16)) * np.random.uniform(*ebrick_chiplets_range), -2)
        ebrick_chiplets_licensing = round(np.random.uniform(*ebrick_chiplets_licensing_range), -2)
        osat = round(np.random.uniform(*osat_range), -2)
        vv_tests = round(np.random.uniform(*vv_tests_range), -2)
        cost_before_profit = round((overhead + cots_chips + custom_chips + custom_chips_nre +
                              custom_chips_licensing + ebrick_chiplets + ebrick_chiplets_licensing +
                              osat +vv_tests), -2)
        profit = round(np.random.uniform(profit_margin_range[0]/100, profit_margin_range[1]/100) * cost_before_profit, -2)
        total_cost = round(cost_before_profit + profit, -2)

        df = df.append({
            'Overhead': overhead,
            'COTS Chips': cots_chips,
            'Custom Chips': custom_chips,
            'Custom Chips NRE': custom_chips_nre,
            'Custom Chips Licensing': custom_chips_licensing,
            'eBrick Chiplets': ebrick_chiplets,
            'eBrick Chiplets Licensing': ebrick_chiplets_licensing,
            'OSAT': osat,
            'V&V Tests': vv_tests,
            'Profit': profit,
            'Total Cost': total_cost
        }, ignore_index=True)
    return df

df = simulate(num_simulations, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range)

# Plot the histogram of total costs
st.subheader('Histogram of Total Costs')
fig = px.histogram(df, x='Total Cost', nbins=50, marginal='box')
st.plotly_chart(fig)

# Identify the largest cost drivers
st.subheader('Largest Cost Drivers')
cost_drivers = df.drop(columns='Total Cost').mean().sort_values(ascending=False)
st.write(cost_drivers)

# Identify the ideal value range for each variable to bring the average total cost below $5M
st.subheader('Ideal Value Range for Each Variable')
for column in df.columns:
    if column != 'Total Cost':
        ideal_range = df[df['Total Cost'] < 5e6][column].agg(['min', 'max'])
        st.write(f'{column}: {ideal_range[0]} - {ideal_range[1]}')

# Identify the profit margin needed to keep the average total cost below $5M
st.subheader('Profit Margin Needed')
profit_margin_needed = df[df['Total Cost'] < 5e6]['Profit'].mean() / df[df['Total Cost'] < 5e6].drop(columns='Profit').sum(axis=1).mean()
st.write(f'Profit margin needed to keep the average total cost below $5M: {profit_margin_needed * 100}%')

# Downloadable results
if st.button('Download Results as CSV'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="simulation_results.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
