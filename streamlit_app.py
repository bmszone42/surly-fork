import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64

# Sidebar inputs
num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1000, max_value=100000, value=10000)
num_bins = st.sidebar.number_input('Number of Bins', min_value=10, max_value=100, value=50)

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

run_simulation = st.button('Run Simulation')

# Perform the simulations
@st.cache
def simulate(num_simulations, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range):
    simulation_data = []  # Create an empty list to store simulation data
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
                              osat + vv_tests), -2)
        profit = round(np.random.uniform(profit_margin_range[0]/100, profit_margin_range[1]/100) * cost_before_profit, -2)
        total_cost = round(cost_before_profit + profit, -2)

        simulation_data.append({
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
        })

    df = pd.DataFrame(simulation_data)
    return df

def reset_data():
    # Reset all the inputs and the dataframe
    num_simulations = 10000
    num_bins = 50
    overhead_range = (2000, 200000)
    cots_chips_range = (1000, 10000)
    custom_chips_range = (1000, 10000)
    custom_chips_nre_range = (1000000, 10000000)
    custom_chips_licensing_range = (0, 1000000)
    ebrick_chiplets_range = (20, 150)
    ebrick_chiplets_licensing_range = (0, 1000000)
    osat_range = (500000, 750000)
    vv_tests_range = (500000, 750000)
    profit_margin_range = (20, 30)
    df = pd.DataFrame()

# Check if the Reset button is clicked
if st.button('Reset'):
    reset_data()

# Check if the Run Simulation button is clicked
if run_simulation:
    # Perform the simulations
    df = simulate(num_simulations, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range)

    # Data Summary
    st.subheader('Data Summary')
    cost_summary = df['Total Cost'].describe()
    st.write(cost_summary)

    # Box Plot of Total Cost
    st.subheader('Box Plot of Total Cost')
    fig_boxplot = px.box(df, y='Total Cost')
    st.plotly_chart(fig_boxplot)

    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    correlation_matrix = df.drop(columns=['Profit', 'Total Cost']).corr()
    fig_heatmap = px.imshow(correlation_matrix)
    st.plotly_chart(fig_heatmap)

    # Pairwise Scatter Plot
    st.subheader('Pairwise Scatter Plot')
    fig_scatter = px.scatter_matrix(df, dimensions=df.columns[:-2])
    st.plotly_chart(fig_scatter)

    # Data Filtering
    st.subheader('Data Filtering')
    profit_margin_filter = st.slider('Filter by Profit Margin', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    filtered_df = df[df['Profit'] / df['Total Cost'] >= profit_margin_filter]
    st.write(filtered_df)

# Downloadable results
if st.button('Download Results as CSV'):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="simulation_results.csv">Download CSV File</a>'
    st.markdown(href, unsafe_allow_html=True)
