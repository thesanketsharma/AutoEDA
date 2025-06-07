import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import sweetviz
from ydata_profiling import ProfileReport
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
import ipywidgets as widgets
import numpy as np
import warnings

warnings.filterwarnings("ignore")


class AutoEDA:
    def __init__(self, df: pd.DataFrame, name: str = "Dataset"):
        self.df = df.copy()
        self.name = name
        self.num_cols = self.df.select_dtypes(include='number').columns.tolist()
        self.cat_cols = self.df.select_dtypes(include='object').columns.tolist()

    def overview(self):
        print("Dataset Shape:", self.df.shape)
        print("Data Types:\n", self.df.dtypes)
        print("Missing Values:\n", self.df.isnull().sum())
        print("Summary Statistics:\n", self.df.describe(include='all'))

    def univariate_analysis(self):
        print("\n Univariate Analysis")

        def plot(col):
            if col in self.num_cols:
                fig = px.histogram(self.df, x=col, title=f"Distribution of {col}")
                fig.show()
            elif col in self.cat_cols:
                fig1 = px.bar(self.df[col].value_counts().reset_index(), x=col, y='count',
                             title=f"Count of {col}")
                fig1.show()
                fig2 = px.pie(self.df[col].value_counts().reset_index(), values='count',names=col, title=f"Proportion of {col}")
                fig2.show()

        dropdown = widgets.Dropdown(options=self.df.columns.tolist(), description="Select Feature")
        ui = widgets.VBox([dropdown])
        out = widgets.interactive_output(plot, {'col': dropdown})
        display(ui, out)

    def bivariate_analysis(self):
        print("Bivariate Analysis")
        """if not target or target not in self.df.columns:
            print("Please provide a valid target column.")
            return"""

        def plot(target, x_col):
            if x_col == target:
                return
            if target in self.num_cols and x_col in self.num_cols:
                fig = px.scatter(self.df, x=x_col, y=target, title=f"{x_col} vs {target}")
                fig.show()
            elif target in self.cat_cols and x_col in self.num_cols:
                fig = px.box(self.df, x=target, y=x_col, title=f"{target} vs {x_col}")
                fig.show()

        dropdown1 = widgets.Dropdown(options=[col for col in self.df.columns], description="Target_column")
        dropdown2 = widgets.Dropdown(options=[col for col in self.df.columns], description="Compare with")
        ui = widgets.VBox([dropdown1, dropdown2])
        out = widgets.interactive_output(plot, {'target': dropdown1, 'x_col': dropdown2})
        display(ui, out)

    def correlation_heatmap(self):
        if len(self.num_cols) < 2:
            print("Not enough numerical columns for correlation heatmap.")
            return
        print("\n🔗 Correlation Heatmap")
        corr = self.df[self.num_cols].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.show()

    def detect_outliers(self, method='zscore', threshold=3):
        print("\n⚠️ Outlier Detection")
        outlier_info = {}

        for col in self.num_cols:
            if method == 'zscore':
                z_scores = zscore(self.df[col].dropna())
                outliers = (np.abs(z_scores) > threshold).sum()
            elif method == 'iqr':
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((self.df[col] < q1 - 1.5 * iqr) | (self.df[col] > q3 + 1.5 * iqr)).sum()
            outlier_info[col] = outliers

        for col, count in outlier_info.items():
            print(f"{col}: {count} outliers")

    def analyze_time_series(self, datetime_col=None, value_col=None, freq='D'):
        print("\n⏳ Time-Series Decomposition")

        if not datetime_col or not value_col:
            print("Please specify a datetime column and a value column.")
            return

        if datetime_col not in self.df.columns or value_col not in self.df.columns:
            print("Invalid columns provided.")
            return

        df_ts = self.df[[datetime_col, value_col]].dropna()
        df_ts[datetime_col] = pd.to_datetime(df_ts[datetime_col])
        df_ts = df_ts.set_index(datetime_col).sort_index()
        df_ts = df_ts.asfreq(freq)

        decomposition = seasonal_decompose(df_ts[value_col], model='additive')

        decomposition.plot()
        plt.tight_layout()
        plt.show()

    def generate_profile_report(self, output_file="profile_report.html"):
        print("\n📑 Generating Pandas Profiling Report...")
        profile = ProfileReport(self.df, title=f"{self.name} Report", explorative=True)
        profile.to_file(output_file)
        print(f"Report saved to: {output_file}")

    def generate_sweetviz_report(self, output_file="sweetviz_report.html"):
        print("\n🍭 Generating Sweetviz Report...")
        report = sweetviz.analyze(self.df)
        report.show_html(output_file)
        print(f"Report saved to: {output_file}")
