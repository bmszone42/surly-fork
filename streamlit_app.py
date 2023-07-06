import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import scipy.stats as stats
import time

def get_input_parameters():
    # Sidebar inputs
    num_simulations = st.sidebar.number_input('Number of Simulations', min_value=1000, max_value=100000, value=10000)

    # User input for cost ranges
    st.sidebar.header('Cost Ranges')
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

    return num_simulations, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range

def get_scenario_parameters():
    # User defines two scenarios
    st.sidebar.header('Scenario 1')
    overhead_range1 = st.sidebar.slider('Overhead Range 1 ($)', min_value=2000, max_value=200000, value=(2000, 200000))
    cots_chips_range1 = st.sidebar.slider('COTS Chips Range 1 ($)', min_value=1000, max_value=10000, value=(1000, 10000))

    st.sidebar.header('Scenario 2')
    overhead_range2 = st.sidebar.slider('Overhead Range 2 ($)', min_value=2000, max_value=200000, value=(2000, 200000))
    cots_chips_range2 = st.sidebar.slider('COTS Chips Range 2 ($)', min_value=1000, max_value=10000, value=(1000, 10000))

    return overhead_range1, cots_chips_range1, overhead_range2, cots_chips_range2

@st.cache
def simulate(num_simulations, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range):
    data = []
    for _ in range(num_simulations):
        overhead = np.random.uniform(*overhead_range)
        cots_chips = np.random.randint(1, 6) * np.random.uniform(*cots_chips_range)
        custom_chips = np.random.randint(0, 3) * np.random.uniform(*custom_chips_range)
        custom_chips_nre = np.random.uniform(*custom_chips_nre_range)
        custom_chips_licensing = np.random.uniform(*custom_chips_licensing_range)
        ebrick_chiplets = np.random.choice(np.arange(16, 257, 16)) * np.random.uniform(*ebrick_chiplets_range)
        ebrick_chiplets_licensing = np.random.uniform(*ebrick_chiplets_licensing_range)
        osat = np.random.uniform(*osat_range)
        vv_tests = np.random.uniform(*vv_tests_range)
        cost_before_profit = (overhead + cots_chips + custom_chips + custom_chips_nre +
                              custom_chips_licensing + ebrick_chiplets + ebrick_chiplets_licensing +
                              osat + vv_tests)
        profit = np.random.uniform(profit_margin_range[0]/100, profit_margin_range[1]/100) * cost_before_profit
        total_cost = cost_before_profit + profit

        data.append({
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
    df = pd.DataFrame(data)
    return df

def compare_scenarios(df1, df2):
    # Compare results
    st.header('Scenario Comparison')
    st.write(f'Mean total cost for scenario 1: {df1["Total Cost"].mean()}')
    st.write(f'Mean total cost for scenario 2: {df2["Total Cost"].mean()}')

    # Calculate 95% confidence interval for mean total cost
    mean1 = df1['Total Cost'].mean()
    std_err1 = df1['Total Cost'].sem()
    ci1 = stats.t.interval(0.95, df1.shape[0]-1, loc=mean1, scale=std_err1)

    mean2 = df2['Total Cost'].mean()
    std_err2 = df2['Total Cost'].sem()
    ci2 = stats.t.interval(0.95, df2.shape[0]-1, loc=mean2, scale=std_err2)

    st.write(f'95% confidence interval for mean total cost in scenario 1: {ci1}')
    st.write(f'95% confidence interval for mean total cost in scenario 2: {ci2}')

def plot_scatter(df):
    # User selects two variables
    var1 = st.sidebar.selectbox('Variable 1', df.columns)
    var2 = st.sidebar.selectbox('Variable 2', df.columns)

    # Plot scatter plot of the two variables
    fig = px.scatter(df, x=var1, y=var2)
    st.plotly_chart(fig)

def plot_histogram(df):
    # User selects a variable
    var = st.sidebar.selectbox('Variable for histogram', df.columns)
    # Plot histogram of the selected variable
    fig = px.histogram(df, x=var)
    st.plotly_chart(fig)

def plot_boxplot(df):
    # User selects a variable
    var = st.sidebar.selectbox('Variable for boxplot', df.columns)
    # Plot boxplot of the selected variable
    fig = px.box(df, y=var)
    st.plotly_chart(fig)


def download_results(df):
    # Downloadable results
    if st.button('Download Results as CSV'):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="simulation_results.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)

def main():
    num_simulations, overhead_range, cots_chips_range, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range = get_input_parameters()
    overhead_range1, cots_chips_range1, overhead_range2, cots_chips_range2 = get_scenario_parameters()

    if st.button('Run Simulation'):
        start_time = time.time()
        # Run simulation for each scenario
        df1 = simulate(num_simulations, overhead_range1, cots_chips_range1, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range)
        df2 = simulate(num_simulations, overhead_range2, cots_chips_range2, custom_chips_range, custom_chips_nre_range, custom_chips_licensing_range, ebrick_chiplets_range, ebrick_chiplets_licensing_range, osat_range, vv_tests_range, profit_margin_range)
        end_time = time.time()
        st.write(f'Simulation run time: {end_time - start_time} seconds')

        compare_scenarios(df1, df2)
        plot_scatter(df1)
        plot_histogram(df1)
        plot_boxplot(df1)
        download_results(df1)

    if st.button('Reset Simulation'):
        st.experimental_rerun()

if __name__ == "__main__":
    main()
