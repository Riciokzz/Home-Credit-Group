import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc


plot_size = (14, 6)


def plot_config():
    """
    Show plot and apply tight layout.
    """
    plt.show()
    plt.tight_layout()


def pie_plot(df: pd.DataFrame, target_col: str):
    """
    Plot pie plot.

    Parameters:
        df: Data dataframe.
        target_col:  Target feature.
    """
    data = df[target_col].value_counts()

    plt.figure(figsize=plot_size)
    plt.pie(data, labels=data.index, autopct="%1.1f%%", explode=[0.02, 0.02])
    plt.title(f"Pie Chart: {target_col}")

    plot_config()


def feature_balance(df: pd.DataFrame, target_col: str, hue: str = None) -> None:
    """
    Plot feature imbalance.

    Parameters:
    df: Data dataframe.
    target_col:  Target feature.
    hue:  Target feature.
    """
    plt.figure(figsize=plot_size)
    ax = sns.countplot(x=target_col, data=df, hue=hue)

    for container in ax.containers:
        ax.bar_label(container, fmt=lambda v: f"{v / len(df) * 100:.2f}%")

    if hue in df.columns:
        ax.set_title(f"{target_col} Distribution Over {hue}")
    else:
        ax.set_title(f"{target_col} Distribution")

    ax.set_ylabel("COUNT")

    plot_config()


def horizontal_kde_box_plot(df: pd.DataFrame, x: str, hue: str) -> None:
    """
    Plot horizontal kde plot.
    Parameters:
        df: Data dataframe.
        x:  Target feature.
        hue:  Target feature.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)

    sns.boxplot(data=df, x=x, hue=hue, ax=ax1)
    sns.kdeplot(data=df, x=x, hue=hue, ax=ax2)

    ax1.set_title(f"{x} Distribution Over {hue}")
    ax2.set_title(f"{x} Denstity")
    ax2.set_xlim(df[x].min() - 0.1)

    plot_config()


def plot_percentage_bars_numerical(
    data: pd.DataFrame, x_feature: str, target_feature: str, max_values: int = 3
) -> None:
    """
    Plot percentage bars for a numerical feature grouped by target feature.

    Parameters:
    data (pd.DataFrame): Input DataFrame.
    x_feature (str): Numerical feature to group.
    target_feature (str): Target feature to plot percentages.
    max_values (int): Maximum number of groups to show, defaults to 3.
    """

    # Create a custom grouping for the x_feature based on max_values
    data = data.copy()
    data["GROUPS"] = data[x_feature].apply(
        lambda x: "0" if x == 0 else str(x) if x < max_values else f"More than {max_values - 1}"
    )

    # Calculate counts and percentages
    feature_target = data.groupby(["GROUPS", target_feature]).size().reset_index(name="count")
    feature_total = data.groupby("GROUPS").size().reset_index(name="total")
    feature_target = feature_target.merge(feature_total, on="GROUPS")
    feature_target["percent"] = (feature_target["count"] / feature_target["total"]) * 100

    # Ensure correct order of groups for plotting
    unique_values = sorted(feature_target["GROUPS"].unique(), key=lambda x: (x[0] == ">", x))

    # Create the bar plot
    ax = sns.barplot(data=feature_target, x="GROUPS", y="percent", hue=target_feature, order=unique_values)

    # Annotate each bar with its percentage value using ax.bar_label
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", label_type="edge")

    # Set labels and title
    plt.ylabel("Percentage")
    plt.title(f"Percentage of {target_feature} within Each {x_feature} Group")
    plt.show()
    plot_config()


def plot_percentage_bars_categorical(
        data: pd.DataFrame, x_feature: str, target_feature: str, max_values=3
) -> None:
    # Count occurrences of each category in x_feature
    counts = data[x_feature].value_counts()

    # If the number of unique categories is less than or equal to max_values,
    # create a list of all categories
    if len(counts) <= max_values:
        top_categories = counts.index.tolist()
        top_counts = counts.tolist()
    else:
        # Get the top categories and sum others
        top_categories = counts.nlargest(max_values).index.tolist()
        other_count = counts[~counts.index.isin(top_categories)].sum()

        # Create a new DataFrame for plotting with top categories plus 'Others'
        top_counts = counts[top_categories].tolist()
        top_counts.append(other_count)  # Add count for 'Others'
        top_categories.append('Others')  # Add 'Others' category

    # Prepare a DataFrame for the top categories
    plot_data = pd.DataFrame({
        x_feature: top_categories,
        'count': top_counts
    })

    # Ensure x_feature is not a categorical type to allow adding 'Others'
    if pd.api.types.is_categorical_dtype(data[x_feature]):
        data[x_feature] = data[x_feature].astype(str)

    # Replace lesser categories with 'Others'
    merged_data = data.copy()
    if len(counts) > max_values:
        merged_data[x_feature] = merged_data[x_feature].where(
            merged_data[x_feature].isin(top_categories[:-1]), 'Others'
        )

    # Group by the new x_feature and target_feature to calculate counts
    feature_target = merged_data.groupby([x_feature, target_feature]).size().reset_index(name="count")

    # Calculate total counts for each category in feature_target
    feature_total = feature_target.groupby(x_feature)['count'].sum().reset_index(name="total")

    # Merge counts to calculate percentages
    feature_target = feature_target.merge(feature_total, on=x_feature)
    feature_target["percent"] = (feature_target["count"] / feature_target["total"]) * 100

    # If the new categories do not match the prepared plot_data, merge to keep consistent categories
    plot_data = plot_data.merge(feature_target[[x_feature, 'percent']], on=x_feature, how='left').fillna(0)

    # Sort feature_target by count for ordering the bars in the plot
    feature_target = feature_target.sort_values(by='count', ascending=False)

    # Create the bar plot
    ax = sns.barplot(data=feature_target, x=x_feature, y='percent', hue=target_feature, order=feature_target[x_feature])

    # Annotate each bar with its percentage value using ax.bar_label
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f%%", label_type="edge")

    # Set labels and title
    plt.ylabel("Percentage")
    plt.title(f"Percentage of {target_feature} within Each {x_feature} Group")
    plot_config()


def cm_matrix(cm: np.ndarray, place: int, model_name: str, axes: np.ndarray) -> None:
    """
    Plots a confusion matrix heatmap.

    Parameters:
        cm: Confusion matrix (2D array-like).
        place: Index of the subplot position.
        model_name: Name of the model for the title.
        axes: Array of axes from plt.subplots.
    """
    # Flatten the axes to handle 2D grid
    flat_axes = axes.flatten()

    sns.heatmap(cm, annot=True, cmap="Blues", fmt=".3g", ax=flat_axes[place], cbar=True)
    flat_axes[place].set_title(f"{model_name}")
    flat_axes[place].set_xlabel("Predicted Label")
    flat_axes[place].set_ylabel("True Label")

    plot_config()


def plot_roc(y_val: pd.Series, y_pred: pd.Series) -> None:
    """
    Plot ROC curve plot.

    Parameters:
    - y_val (Series): True labels.
    - y_pred (Series): Predicted probabilities for the positive class.
    """
    fpr, tpr, _ = roc_curve(y_val, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr, tpr, color="darkorange", label="ROC curve (area = {:.2f})".format(roc_auc)
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("(ROC) Curve")
    plt.legend(loc="lower right")
    plot_config()
