import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.ML_PROJECT.pipeline.prediction_pipeline import (
    PredictPipeline,
    CustomData
)


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Insurance Charges Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    data_path = os.path.join("artifacts", "clean_data.csv")
    return pd.read_csv(data_path)


df = load_data()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.title("💰 Insurance ML")
    st.write("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Dataset Overview",
            "📈 Data Visualization",
            "🤖 Model Information",
            "💰 Predict Charges",
            "📉 Model Performance",
            "ℹ️ About"
        ]
    )

    st.write("---")
    st.caption("Insurance Charges Prediction System")
    st.caption("Machine Learning Project")


# ==================================================
# HOME PAGE
# ==================================================
if page == "🏠 Home":

    st.title("💰 Insurance Charges Prediction System", anchor=False)

    st.write(
        """
        Welcome to the **Insurance Charges Prediction System**.

        This Machine Learning application predicts estimated medical
        insurance charges based on a person's age, BMI, children,
        gender, smoking status and region.
        """
    )

    st.write("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📊 Total Records",
            df.shape[0]
        )

    with col2:
        st.metric(
            "🔢 Total Features",
            df.shape[1] - 1
        )

    with col3:
        st.metric(
            "💰 Average Charges",
            f"${df['charges'].mean():,.0f}"
        )

    with col4:
        st.metric(
            "📈 Maximum Charges",
            f"${df['charges'].max():,.0f}"
        )

    st.write("---")

    st.subheader("🎯 Project Objective", anchor=False)

    st.info(
        "The main objective of this project is to predict medical "
        "insurance charges using Machine Learning regression models."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📥 Input Features", anchor=False)

        st.write("""
        - Age
        - BMI
        - Number of Children
        - Gender
        - Smoking Status
        - Region
        """)

    with col2:
        st.subheader("💰 Output", anchor=False)

        st.success(
            "Predicted Medical Insurance Charges"
        )


# ==================================================
# DATASET OVERVIEW
# ==================================================
elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview", anchor=False)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    st.write("---")

    st.subheader("🔍 Dataset Preview", anchor=False)
    st.dataframe(df.head(10), use_container_width=True)

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📌 Data Types", anchor=False)
        st.dataframe(
            df.dtypes.astype(str),
            use_container_width=True
        )

    with col2:
        st.subheader("❓ Missing Values", anchor=False)
        missing_values = df.isnull().sum()
        st.dataframe(
            missing_values,
            use_container_width=True
        )

    st.write("---")

    st.subheader("📈 Statistical Summary", anchor=False)
    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ==================================================
# DATA VISUALIZATION
# ==================================================
elif page == "📈 Data Visualization":

    st.title("📈 Data Visualization", anchor=False)

    graph = st.selectbox(
        "Select Visualization",
        [
            "Charges Distribution",
            "Age vs Charges",
            "BMI vs Charges",
            "Smoker vs Charges",
            "Children vs Charges",
            "Correlation Heatmap"
        ]
    )

    if graph == "Charges Distribution":

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(
            df["charges"],
            kde=True,
            ax=ax
        )

        ax.set_title("Distribution of Insurance Charges")
        ax.set_xlabel("Charges")

        st.pyplot(fig)


    elif graph == "Age vs Charges":

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(
            data=df,
            x="age",
            y="charges",
            ax=ax
        )

        ax.set_title("Age vs Insurance Charges")

        st.pyplot(fig)


    elif graph == "BMI vs Charges":

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.scatterplot(
            data=df,
            x="bmi",
            y="charges",
            ax=ax
        )

        ax.set_title("BMI vs Insurance Charges")

        st.pyplot(fig)


    elif graph == "Smoker vs Charges":

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="smoker_yes",
            y="charges",
            ax=ax
        )

        ax.set_title("Smoker vs Insurance Charges")

        st.pyplot(fig)


    elif graph == "Children vs Charges":

        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=df,
            x="children",
            y="charges",
            ax=ax
        )

        ax.set_title("Children vs Insurance Charges")

        st.pyplot(fig)


    elif graph == "Correlation Heatmap":

        fig, ax = plt.subplots(figsize=(10, 7))

        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            ax=ax
        )

        ax.set_title("Correlation Heatmap")

        st.pyplot(fig)


# ==================================================
# MODEL INFORMATION
# ==================================================
elif page == "🤖 Model Information":

    st.title("🤖 Machine Learning Model", anchor=False)

    st.subheader("Model Training Process", anchor=False)

    st.write(
        """
        Different regression models were trained and compared.
        The best performing model was selected and saved.
        """
    )

    st.write("---")

    st.subheader("🤖 Models Compared", anchor=False)

    models_data = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Decision Tree Regressor",
            "Random Forest Regressor",
            "KNeighbors Regressor",
            "Support Vector Regressor",
            "XGBoost Regressor"
        ],
        "Purpose": [
            "Baseline Regression Model",
            "Tree Based Regression",
            "Ensemble Regression",
            "Neighbour Based Regression",
            "Support Vector Regression",
            "Gradient Boosting Regression"
        ]
    })

    st.dataframe(
        models_data,
        use_container_width=True
    )

    st.write("---")

    st.subheader("🏆 Final Model", anchor=False)

    st.success(
        "🏆 Random Forest Regressor was selected as the final model "
        "with an R² Score of 0.6141."
    )

    st.code(
        "models/train_model.pkl"
    )

    st.subheader("⚙️ Data Scaler", anchor=False)

    st.info(
        "The trained scaler is saved as scaler.pkl and is used to "
        "transform user input before prediction."
    )

    st.code(
        "models/scaler.pkl"
    )


# ==================================================
# PREDICTION PAGE
# ==================================================
elif page == "💰 Predict Charges":

    st.title("💰 Predict Insurance Charges", anchor=False)

    st.write(
        "Enter the details below and click the prediction button."
    )

    st.write("---")

    col1, col2 = st.columns(2)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=25,
            step=1
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        children = st.number_input(
            "Number of Children",
            min_value=0,
            max_value=10,
            value=0,
            step=1
        )


    with col2:

        sex = st.selectbox(
            "Sex",
            ["Female", "Male"]
        )

        smoker = st.selectbox(
            "Smoker",
            ["No", "Yes"]
        )

        region = st.selectbox(
            "Region",
            [
                "Northeast",
                "Northwest",
                "Southeast",
                "Southwest"
            ]
        )


    st.write("")

    if st.button(
        "🔮 Predict Insurance Charges",
        use_container_width=True
    ):

        sex_male = 1 if sex == "Male" else 0

        smoker_yes = 1 if smoker == "Yes" else 0

        region_northwest = 1 if region == "Northwest" else 0
        region_southeast = 1 if region == "Southeast" else 0
        region_southwest = 1 if region == "Southwest" else 0


        data = CustomData(
            age=age,
            bmi=bmi,
            children=children,
            sex_male=sex_male,
            smoker_yes=smoker_yes,
            region_northwest=region_northwest,
            region_southeast=region_southeast,
            region_southwest=region_southwest
        )


        input_df = data.get_data_as_dataframe()


        predict_pipeline = PredictPipeline()

        prediction = predict_pipeline.predict(
            input_df
        )


        predicted_value = prediction[0]


        st.write("---")

        st.success("Prediction completed successfully! 🎉")

        col1, col2, col3 = st.columns(3)

        with col2:
            st.metric(
                "💰 Estimated Insurance Charges",
                f"${predicted_value:,.2f}"
            )

        st.write("---")

        st.subheader("Prediction Summary", anchor=False)

        result_col1, result_col2, result_col3 = st.columns(3)

        with result_col1:
            st.info(f"👤 Age: {age}")

        with result_col2:
            st.info(f"📊 BMI: {bmi}")

        with result_col3:
            st.info(f"🚬 Smoker: {smoker}")

# ==================================================
# MODEL PERFORMANCE
# ==================================================
elif page == "📉 Model Performance":

    st.title("📉 Model Performance", anchor=False)

    st.subheader("🏆 Best Model", anchor=False)

    st.success(
        "Random Forest Regressor was selected as the best model "
        "based on the highest R² Score."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🏆 Best Model",
            "Random Forest"
        )

    with col2:
        st.metric(
            "📈 R² Score",
            "0.6141"
        )

    st.write("---")

    st.subheader("📊 Model Comparison", anchor=False)

    model_results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "KNN",
            "SVR",
            "Decision Tree",
            "Random Forest",
            "XGBoost Regression"
        ],
        "Best Score": [
            0.5996,
            0.5828,
            0.0439,
            0.5785,
            0.6469,
            0.5591
        ],
        "R² Score": [
            0.5733,
            0.5160,
            0.0266,
            0.4046,
            0.6141,
            0.1204
        ]
    })

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    st.write("---")

    st.subheader("📈 R² Score Comparison", anchor=False)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        model_results["Model"],
        model_results["R² Score"]
    )

    ax.set_xlabel("Models")
    ax.set_ylabel("R² Score")
    ax.set_title("R² Score Comparison of Regression Models")

    plt.xticks(rotation=25)

    st.pyplot(fig)

    st.write("---")

    st.subheader("🔍 Best Model Details", anchor=False)

    st.info(
        """
        Random Forest Regressor achieved the highest R² Score of 0.6141.
        Therefore, it was selected as the final model and saved for
        insurance charges prediction.
        """
    )

    st.code(
        "models/train_model.pkl"
    )

# ==================================================
# ABOUT PAGE
# ==================================================
elif page == "ℹ️ About":

    st.title("ℹ️ About Project", anchor=False)

    st.subheader(
        "Insurance Charges Prediction Using Machine Learning",
        anchor=False
    )

    st.write("---")

    st.subheader("🎯 Objective", anchor=False)

    st.write(
        """
        The objective of this project is to develop a Machine Learning
        application capable of predicting medical insurance charges
        based on personal and health-related information.
        """
    )

    st.subheader("🛠️ Technologies Used", anchor=False)

    st.write("""
    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Scikit-learn
    - Streamlit
    - Joblib
    """)

    st.subheader("📁 Project Components", anchor=False)

    st.write("""
    - Data Cleaning
    - Exploratory Data Analysis
    - Machine Learning Model Training
    - Model Comparison
    - Model Selection
    - Model Saving
    - Prediction Pipeline
    - Streamlit Deployment
    """)

    st.success(
        "Developed as a Machine Learning Final Project 🚀"
    )