
#Streamlit Dummy Dataset Creator - v3 
#Written under duress by James Craft esq.

#lets check to see if you're setup correctly...
import subprocess

# List of required packages
required_packages = ['pandas', 'plotly', 'streamlit']

# Function to check if a package is installed
def is_package_installed(package_name):
    try:
        subprocess.check_output(['pip', 'show', package_name])
        return True
    except subprocess.CalledProcessError:
        return False

# Check and install missing packages
for package in required_packages:
    if not is_package_installed(package):
        print(f"Installing {package}...")
        subprocess.call(['pip', 'install', package])

#lets get lit. 
#Streamlit that is..
#i'll show myself out..
import os
import streamlit as st
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px 
import base64
from dateutil.relativedelta import relativedelta

def calculate_months_difference(start_date, end_date):
    # Calculate the difference in months using relativedelta
    delta = relativedelta(end_date, start_date)

    # Calculate the total months difference
    months_difference = delta.years * 12 + delta.months

    return months_difference

#The below function is what we'll call once all of the argument are disclosed/can be passed. 
#This function is responsible for creating the dataset based upon the streamlit code we have at the bottom.

#TLDR; we create some time sets, cycle through the product cycling through each product populate a dataset with entirely random values from our given ranges.
#we then create a ratio for each day  

def generate_dummy_dataset(date_range, product_list, dimension_list, metric_ranges, rate_of_improvement, calculated_metrics_list, ranking_values):
    # Extract start and end dates from the date_range list
    start_date, end_date = date_range
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    total_months = calculate_months_difference(start_date, end_date)
    # Create an empty list to store the data
    data = []
    
    #lets start at the start eh!
    current_date = start_date
    #
    while current_date <= end_date:
    #willing we've not exceeded our end date in our loop
        for product in product_list:
       #we cycle through products
            # Randomly selecting the metric values within our specified range 
            metric_values = {metric: random.uniform(metric_ranges[metric][0], metric_ranges[metric][1])
                for metric in metric_ranges }

  # Adjust metric values depending upon on the preselected rate of improvement and number of months remaining this is adjusted linearly :( sorry i'm not that good it's still 2 layers deep mind..
            months_passed = (current_date.year - start_date.year) * 12 + (current_date.month - start_date.month)
            Perc_passed = months_passed / total_months
            for metric in rate_of_improvement:
            #use the below to determine % of way through to end rate
                start_rate, end_rate = rate_of_improvement[metric]
                #improvement_ratio = start_rate + (months_passed / ((end_date.year - start_date.year) * 12 + (end_date.month - start_date.month))) * (end_rate - start_rate)
                
                if metric in metric_values:
                # checks if we've selected negative change
                    if switch_negative_improvement:
                    #swaps percents to -% values
                            end_rate = -abs(end_rate)
                            start_rate = -abs(start_rate)
                            #creates an average as a mid ground between start and end
                            mean_rate = (end_rate+start_rate)/2
                            #if we're below 25% completion use start rate
                            if Perc_passed < 0.25:
                                improvement_ratio = start_rate
                            #if we're between 25 and 75% completion use the average
                            elif Perc_passed > 0.25 and Perc_passed < 0.75:
                                improvement_ratio = mean_rate
                            #if we're greater than 75% of the way through then use the end rate
                            elif Perc_passed > 0.76:
                                improvement_ratio = end_rate
                            else: improvement_ratio = mean_rate
                            metric_values[metric] *= improvement_ratio
                  #create the final metric values based upon our improvement ratio - *= multiply and - so metric * by our improvement ratio becomes the new value for metric values... if that makes any sense let me know :') if not let me know all the same :')
                    else:
                    #if we're positive then do teh same as the above but using start/end rates as is
                        mean_rate = np.mean(end_rate+start_rate)/2
                        if Perc_passed < 0.25:
                            improvement_ratio = start_rate 
                        elif Perc_passed > 0.25 and Perc_passed < 0.75:
                            improvement_ratio = mean_rate
                        elif Perc_passed > 0.76:
                            improvement_ratio = end_rate
                        else: 
                            improvement_ratio = mean_rate
                            metric_values[metric] *= improvement_ratio
                
                # Apply variations based on rankings
                ranking_value = ranking_values.get(product, 1)
                # Adjust metric values based on ranking
                for metric in metric_values:
                    metric_values[metric] *= ranking_value

            # Append the instance for the current month, metric and product to the data we initalised on L.27
            data.append([current_date.strftime('%Y-%m-%d'), product] + [metric_values[metric] for metric in dimension_list[2:]])

        # if we're in decemeber, add 1 to the year.. HNY! and pop us in jan. if not.. move us on a month.
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    # Convert the data list to a pandas DataFrame because is jummier.
    df = pd.DataFrame(data, columns=dimension_list)

    # Check if there are any calculated metrics, if so.. is it empty *upside down smiley*? if not split it by the ='s create a new column by the first item we split by, then eval the second *crosses fingers* - this bit could do with a bit of dev work imho
    if len(calculated_metrics_list) != 0:
        for metric_formula in calculated_metrics_list:
            if metric_formula.strip() != '':
             #create a name from the left part of the ='s and the right becomes our column
                metric_name, formula = metric_formula.split('=')
                df[metric_name.strip()] = df.eval(formula.strip())
                #Bosh as they say in Essex.
    return df

#########Start of the Streamlit stuff

# Load the background image file 
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"jpg"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )

if "backgroundfile" not in st.session_state:
    st.session_state["backgroundfile"] = round(random.uniform(1,12))

try :
    folder_path = r'C:\Users\james.craft\backgrounds'
except KeyError:
    print("...10,000 years will give you such a crick in the neck")
    whoareu = os.getlogin()
    folder_path = r'C:\Users\{whoareu}\backgrounds'

backgroundfile = folder_path+r"\resized_"+str(st.session_state["backgroundfile"])

#load background randomly from our directory
add_bg_from_local(backgroundfile+".jpg")

# title
st.title('Pitch Data Generator')

# Sidebar
st.sidebar.title('Settings')

# Date range inputs
st.sidebar.subheader('Date Range Entry')
start_date = st.sidebar.date_input('Start Date', datetime(2023, 1, 1))
end_date = st.sidebar.date_input('End Date', datetime(2023, 9, 15))

# product list inputs
st.sidebar.subheader('Product Entry')
product_list = st.sidebar.text_area('Products (comma-separated)', '1, 2, 3, 4').split(',')

#input ranges for metric sliders (min max of the setup)
st.sidebar.subheader('Master Metric Slider Range')
slidermin, slidermax = st.sidebar.slider('slider range', -1000000, 1000000, (-5, 15))

st.sidebar.subheader('Product Ranking Values (Multiplier - Reverse Rank)')
ranking_values = {}
for product in product_list:
    ranking_values[product] = st.sidebar.slider(f'Ranking for {product}', min_value=1, max_value=len(product_list), value=1)
    
sorted_products = sorted(product_list, key=lambda x: ranking_values[x])

# Dimension lists
dimension_list = ['Date', 'product']

#init the metrics range dict
metric_ranges = {}

# metric selection box
st.sidebar.subheader('Metrics')
metric_list = st.sidebar.text_area('Metrics (one per line)', 'Sales\nLeads').splitlines()

for metric in metric_list:
    dimension_list.append(metric)

# metric range sliders
st.sidebar.subheader('Metric Ranges')
for i, metric in enumerate(metric_list):
    # Use a unique key for each slider by adding an index (i) to the key argument
    min_value, max_value = st.sidebar.slider(f'{metric} Range', slidermin, slidermax, (100, 20000), key=f'{metric}_slider_{i}')
    metric_ranges[metric] = (min_value, max_value)

# Rate of improvement inputs
st.sidebar.subheader('Rate of Improvement')
rate_of_improvement = {}
switch_negative_improvement = st.sidebar.checkbox('Use Negative Improvement Rate', value=False)
for metric in metric_list:
    min_rate, max_rate = st.sidebar.slider(f'{metric} Improvement Rate', 0.0, 2.0, (0.5, 1.0), step=0.1)
    rate_of_improvement[metric] = (min_rate, max_rate)
st.sidebar.subheader('Calculated metrics')

# Calculated metrics list
calculated_metrics_list = st.sidebar.text_area('Calculated Metrics (PSV)', 'Leads_to_Sales = Leads / Sales').split('|')

# Generate button that triggers the creation of the summary graphs, table and download link dun kno.
if st.sidebar.button('Generate Data'):
    # Generate the dummy dataset based on user inputs
    dummy_dataset = generate_dummy_dataset(
        (start_date.strftime('%Y-%m-%d'), 
        end_date.strftime('%Y-%m-%d')),
        product_list,
        ['Date', 'product'] + metric_list,
        metric_ranges,
        rate_of_improvement,
        calculated_metrics_list,
        ranking_values
    )

    # Bar graph for total numbers of each metric per product
    st.subheader('Bar Graph: Total Numbers per product')
    st.dataframe(dummy_dataset)
    bar_df = dummy_dataset.groupby('product')[metric_list].sum().sort_values(by=[metric_list][0], ascending=False)
    bar_df = bar_df
    st.bar_chart(bar_df)

    # Line graph for total numbers over time
    st.subheader('Line Graph: Total Numbers over Time')
    line_df = dummy_dataset.groupby('Date')[metric_list].sum()  

    # Using Plotly for the line chart
    fig = px.line(line_df, x=line_df.index, y=metric_list, title='Total Numbers over Time', labels={'x': 'Date', 'value': 'Total Numbers'}, markers=True)
    st.plotly_chart(fig)

    # Show the generated dataset as a table
    st.subheader('Generated Dataset')
    st.dataframe(dummy_dataset)
    
    # Download link/it's functionality for it
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    filename = f'static/dummy_dataset_{timestamp}.csv'
    print(filename)
    dummy_dataset.to_csv(filename, index=False)
    st.markdown(f'[Download dataset]({filename})', unsafe_allow_html=True)

    