##################################################################################
#                                                                                #
# Main                                                                           #
# ::: Handles the navigation / routing and data loading / caching.               #
#                                                                                #
##################################################################################

import pandas as pd
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import eli5
from IPython.display import display
import numpy as np
from streamlit import components
from scipy.spatial import distance

def main():
	'''Set main() function. Includes sidebar navigation and respective routing.'''

	st.sidebar.title("Explore")
	app_mode = st.sidebar.selectbox( "Choose an Action", [
		"About",
        "Data Dictionary",
		"Start Evaluation",

	])

	

	# nav
	if   app_mode == "About": show_about()
	elif app_mode == "Start Evaluation": explore_classified()
	elif app_mode == "Data Dictionary":data_dictionary()


def show_about():
	''' Home page '''
	st.image("https://www.arubanetworks.com/wp-content/uploads/verisk-2-cs.jpg",
             use_column_width=True)
	st.title('Welcome to Home Equity Line of Credit Decision Support System!')
	st.markdown("Last updated on Nov 1 2020") 
	st.subheader("Ya Liu")
	st.write("With this tool, you could quickly get decision support with machine learning techniques. We would provide insights with following methods:")
	st.markdown("1. model's predicted risk performance")
	st.markdown("2. historical data")
	st.markdown("3. rules")

def data_dictionary():
    st.title('What data do I need for risk estimation?')
    st.write("Here is the table for data dictionary. You may refer to this table to understand variable meaning.")
    st.image("https://github.com/yaliu0703/yaliu0703.github.io/blob/master/images/data%20dictionary.png?raw=true",use_column_width=True)
    html_object1 = eli5.show_weights(model, feature_names=list(X_train.columns),top=100)
    raw_html1 = html_object1._repr_html_()
    components.v1.html(raw_html1,height=500,scrolling=True)

##################################################################################
#                                                                                #
# Start Evaluation                                                               #
# ::: Allow the user to pick one row from dataset and evaluate                   #
# ::: and show interpretation                                                    #
#                                                                                #
##################################################################################

def explore_classified():
	# Text on Evaluation page 
	st.title('Home Equity Line of Credit (HELOC) applications decision support system')
	st.write('''
		1. input data 
		2. Click the button "Show evaluation result" to get prediction result and interpretation
	''')

	# Step 1:User chooses one row record from existing FICO credit report dataset
	demo = st.radio('Choose a patient demo',("demo A","demo B"))  # Input the index number
	# Step 2:Get user input
	if demo == "demo A": newdata = get_input(int(3))#bad
	elif demo == "demo B":newdata = get_input(int(37))#good
   
	
	# Step 3: when user checks, run model
	if st.button('Show evaluation result'):
		run_model(newdata)
		
def run_model(newdata):
    st.write(newdata)
    y_pred = model.predict(newdata)
    if y_pred[0] > 0:
        st.text("Risk performance is Bad")
    else:
        st.text("Risk performance is Good")
    st.subheader("Intepretation:")
    html_object = eli5.show_prediction(model, newdata.iloc[0],feature_names=list(newdata.columns), show_feature_values=True)
    raw_html = html_object._repr_html_()
    components.v1.html(raw_html,height=500,scrolling=True)
    find_similar_record(newdata)

def find_similar_record(newdata):
    newdata = newdata[list(df.iloc[:,1:18].columns)]
    newdata_normalized = (newdata - df.iloc[:,1:18].mean()) / df.iloc[:,1:18].std() 
    # Find the distance between lebron james and everyone else.
    euclidean_distances = df_normalized.apply(lambda row: distance.euclidean(row, newdata_normalized), axis=1)
    # Create a new dataframe with distances.
    distance_frame = pd.DataFrame(data={"dist": euclidean_distances, "idx": euclidean_distances.index})
    distance_frame.sort_values("dist", inplace=True) 
    # Find the most similar record to new data 
    second_smallest = distance_frame.iloc[1:4]["idx"]
    most_similar_to_input = df.iloc[second_smallest]
    st.subheader("Here are historical records similar to your input for your reference:")
    st.write(most_similar_to_input)

    
def get_input(index):
    
    values = X_train.iloc[index]  # Input the value from dataset

    # Create input variables for evaluation please use these variables for evaluation
    
    ExternalRiskEstimate = st.sidebar.slider('External Risk Estimate', -9.0, 100.0, float(values[0]))
    MSinceOldestTradeOpen = st.sidebar.text_input('Months Since Oldest Trade Open:', values[1])
    AverageMInFile = st.sidebar.text_input('Average Months in File:', values[2])
    NumSatisfactoryTrades = st.sidebar.text_input('Number of Satisfactory Trades:', values[3])
    PercentTradesNeverDelq = st.sidebar.slider('Percent of Trades Never Delinquent',-10.0, 100.0, float(values[4]))
    MaxDelq2PublicRecLast12M = st.sidebar.slider('Max Deliquncy on Public Records Last 12 Months:', -10.0, 9.0, float(values[5]))
    MaxDelqEver = st.sidebar.slider('Max Deliquncy Ever:', -10.0, 9.0, float(values[6]))
    NumTotalTrades = st.sidebar.text_input('Number of Total Trades', values[7],4)
    NumTradesOpeninLast12M = st.sidebar.text_input('Number of Trades Open in Last 12 Months', values[8],4)
    PercentInstallTrades = st.sidebar.slider('Percent of Installment Trades', -10.0, 100.0, float(values[9]))
    MSinceMostRecentInqexcl7days = st.sidebar.text_input('Months Since Most Recent Inq excl 7days:', values[10])
    NumInqLast6M = st.sidebar.text_input('Number of Inq Last 6 Months:', values[11])
    NetFractionRevolvingBurden = st.sidebar.text_input('Net Fraction Revolving Burden:', values[12])
    NetFractionInstallBurden = st.sidebar.text_input('Net Fraction Installment Burden:', values[13])
    NumRevolvingTradesWBalance = st.sidebar.text_input('Number Revolving Trades with Balance:', values[14])
    NumInstallTradesWBalance = st.sidebar.text_input('Number Installment Trades with Balance:', values[15])
    NumBank2NatlTradesWHighUtilization = st.sidebar.text_input('Number Bank/Natl Trades w high utilization ratio:', values[16])
    PercentTradesWBalance = st.sidebar.slider('Percent Trades with Balance:', -10.0, 100.0, float(values[17]))
    
    newdata = pd.DataFrame()
    newdata = newdata.append({'ExternalRiskEstimate':ExternalRiskEstimate,
            'MSinceOldestTradeOpen':MSinceOldestTradeOpen,
            'AverageMInFile':AverageMInFile, 
            'NumSatisfactoryTrades':NumSatisfactoryTrades,
            'PercentTradesNeverDelq':PercentTradesNeverDelq,
            'MaxDelq2PublicRecLast12M':MaxDelq2PublicRecLast12M,
            'MaxDelqEver':MaxDelqEver,
            'NumTotalTrades':NumTotalTrades,
            "NumTradesOpeninLast12M": NumTradesOpeninLast12M,
            'PercentInstallTrades':PercentInstallTrades,
            'MSinceMostRecentInqexcl7days':MSinceMostRecentInqexcl7days, 
            'NumInqLast6M':NumInqLast6M,
            'NetFractionRevolvingBurden':NetFractionRevolvingBurden, 
            'NetFractionInstallBurden':NetFractionInstallBurden,
            'NumRevolvingTradesWBalance':NumRevolvingTradesWBalance,
            'NumInstallTradesWBalance':NumInstallTradesWBalance,
            'NumBank2NatlTradesWHighUtilization':NumBank2NatlTradesWHighUtilization,
            'PercentTradesWBalance':PercentTradesWBalance},ignore_index=True) 
    newdata = newdata.apply(pd.to_numeric)
    return newdata
        
     
##################################################################################
#                                                                                #
# Execute                                                                        #
#                                                                                #
##################################################################################


if __name__ == "__main__":
	
    X_train = pd.read_csv("X_train.csv")
    y_train = pd.read_csv("y_train.csv")
    df = pd.read_csv("df.csv")
    model = LogisticRegression(C=1.5, class_weight='balanced', random_state=42,solver='liblinear')
    model.fit(X_train, y_train)
    # Normalize all of the numeric columns
    df_normalized = (df.iloc[:,1:18] - df.iloc[:,1:18].mean()) / df.iloc[:,1:18].std()
    

	# execute
    main()


