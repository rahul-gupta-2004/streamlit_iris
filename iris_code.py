import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris

st.set_page_config(
    page_title="Iris Dataset",
    page_icon="🌸",
    layout="wide",
)

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = [iris.target_names[i] for i in iris.target]

avg_stats = df.drop(columns=['species']).mean()

st.title("Iris Species Dashboard")
st.subheader("Key metrics and feature averages for the Fisher's Iris dataset")



with st.container(border=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            label="Avg Sepal Length",
            value=f"{avg_stats['sepal length (cm)']:.2f} cm",
            delta="Overall Mean"
        )
        st.metric(
            label="Avg Petal Length",
            value=f"{avg_stats['petal length (cm)']:.2f} cm",
            delta="Overall Mean"
        )
        
    with col2:
        st.metric(
            label="Avg Sepal Width",
            value=f"{avg_stats['sepal width (cm)']:.2f} cm",
            delta="Overall Mean"
        )
        st.metric(
            label="Avg Petal Width",
            value=f"{avg_stats['petal width (cm)']:.2f} cm",
            delta="Overall Mean"
        )
    
    st.divider()

    bottom_cols = st.columns(2)
    with bottom_cols[0]:
        most_common = df['species'].value_counts().idxmax()
        st.metric("Most Frequent Species", most_common.title())
    
    with bottom_cols[1]:
        st.metric("Total Samples", len(df))

st.write("### Data Preview")
st.dataframe(df.head(10), use_container_width=True)