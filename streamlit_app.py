import streamlit as st
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.linear_model import LinearRegression as LR
from sklearn.metrics import r2_score



data = pd.read_csv('../../data/kc_house_data3.csv',delimiter=";")
X = np.array(data.drop(["prix(dh)"], 1))
Y = np.array(data["prix(dh)"])

st.set_page_config(layout='wide', initial_sidebar_state='expanded') #bash tzid f lcontainer

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Midvi  `DataMining`')

st.sidebar.subheader('Dataset parameter')
SortedBy = st.sidebar.selectbox('Sorted by', ('10','25', '50','100','1000')) 

st.sidebar.subheader('Type of regression')
model_type = st.sidebar.selectbox('Select regression', ('Multiple Lineare regression', 'Polynomial Regression'))
st.sidebar.subheader('Data Visualisation')

visualisation_data = st.sidebar.selectbox('visualiser le prix avec ', ['chambres','bathrooms','surface(m2)','Salon','balcon'])

st.sidebar.subheader('Supprimer Colonnes')
plot_data = st.sidebar.multiselect('Select le colonne a supprimer', ['prix(dh)','chambres','bathrooms','surface(m2)','Salon','balcon'])
button_supp = st.sidebar.button("delete")
if button_supp:
    if plot_data == []:
        st.sidebar.warning("Veuillez choisire les colonnes a supprimer !!")
    else:
        data.drop(plot_data,axis=1, inplace=True)
        st.sidebar.write("Vous avez supprimer avec succes les colonnes ",plot_data)

# plot_height = st.sidebar.slider('Specify plot height', 200, 500, 250)

st.sidebar.markdown('''
---
Created with ❤️ by [EL FEDINI EL MEHDI].
''')


# Row A metric
# st.markdown('### Metrics')
# col1, col2, col3 = st.columns(3)
# col1.metric("Temperature", "70 °F", "1.2 °F")
# col2.metric("Wind", "9 mph", "-8%")
# col3.metric("Humidity", "86%", "4%")

# Row B HeatMap
c1, c2 = st.columns((70,3))
c3, c4 = st.columns((7,3))
with c1:
    st.markdown('### Dataset')
    st.table(data.head(int(SortedBy)))
    if visualisation_data:
        st.markdown('### Data Visualisation')
        plt.rcParams.update({'font.size': 15})
        plt.style.use("dark_background")
        fig = plt.figure(figsize = (15,8))
        fig = plt.figure(figsize = (15,8))
        plt.scatter(data['prix(dh)'],data[visualisation_data])
        plt.xlabel("prix")
        plt.ylabel(visualisation_data)
        # plt.title("Price vs ",fontsize = 20)
        st.pyplot(fig)
# Row C
with c3:
    tabl_val = []
    st.markdown('### Faire une prédiction')

    for i in data:
        tabl_val.append(i)
    for i in range(len(tabl_val)):
        if tabl_val[i] =='prix(dh)':
            st.write("")
        else:
            tabl_val[i] =st.text_input("Entrer {}".format(tabl_val[i]))

    submit = st.button("submit")
with c4:
    st.markdown('### Solution')
    if submit:
        if model_type == 'Multiple Lineare regression':
            # st.warning("Vous aveze entre ")
            # for i in range(len(tabl_val)):
            #     st.write(tabl_val[i])
            regL = LR()
            regL.fit(X,Y)
            R2=r2_score(Y,regL.predict(X))
            n=len(X)
            p=len(X[0,:])
            R2_ajusté=1-(n-1)/(n-1-p)*(1-R2)
            # print(R2_ajusté)
            st.markdown('### Le prix prédicté ')
            st.code(str(regL.predict([[int(tabl_val[1]),int(tabl_val[2]),int(tabl_val[3]),int(tabl_val[4]),int(
                        tabl_val[5])]]))+" Dh")
            st.write("R2_ajusté = ",R2_ajusté)
            # print(type(tabl_val[0]))
            # print(tabl_val[1])
            # print(tabl_val[2])
        elif model_type == 'Polynomial Regression':
            pass
# Define a function to toggle the theme
def toggle_theme():
    is_dark = st.get_option("theme") == "dark"
    st.set_option("theme", "dark" if not is_dark else "light")
    st.experimental_set_query_params(theme="dark" if not is_dark else "light")

# Get the current theme from the URL query parameters
params = st.experimental_get_query_params()
theme = params["theme"][0] if "theme" in params else "light"

# Add a toggle button for the theme
if st.button("Toggle Theme"):
    toggle_theme()

# Wrap the rest of your app in a beta_container with the theme set
with st.container():
    st.write("This is my Streamlit app!")
    st.write("You are currently using the {} theme.".format(theme))

    
# sns.pairplot(data, x_vars=['bedrooms'], y_vars='price', height=4, aspect=1, kind='scatter')
