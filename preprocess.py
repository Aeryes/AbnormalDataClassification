import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path='data/data.csv'):
    # Load data
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Feature selection
    features = ['Source Port', 'Destination Port', 'Protocol', 'Length']
    X = data[features]

    # Convert 'Protocol' to numerical values (if it's categorical)
    X['Protocol'] = X['Protocol'].astype('category').cat.codes

    # Use 'bad_packet' as the label
    y = data['bad_packet']

    return X, y

def analyze_data(data):
    # Display the distribution of 'bad_packet' (normal vs anomaly) using log scale
    print("\nLabel Distribution:")
    label_counts = data['bad_packet'].value_counts()
    print(label_counts)
    label_counts.plot(kind='bar', color=['blue', 'red'], log=True)
    plt.title('Label Distribution: Normal vs Anomaly (Log Scale)')
    plt.xticks(ticks=[0, 1], labels=['Normal', 'Anomaly'], rotation=0)
    plt.ylabel('Number of Samples (Log Scale)')
    plt.show()

    # Feature distributions for each label with normalized histograms
    features = ['Source Port', 'Destination Port', 'Protocol', 'Length']
    for feature in features:
        plt.figure(figsize=(8, 4))
        sns.histplot(
            data=data,
            x=feature,
            hue='bad_packet',
            bins=30,
            kde=True,
            stat='density',  # Normalize the counts
            common_norm=False,  # Normalize each subset separately
            palette={0: 'blue', 1: 'red'}
        )
        plt.title(f'Distribution of {feature} for Normal vs Anomaly (Normalized)')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend(title='Label', labels=['Normal', 'Anomaly'])
        plt.show()

    # Ensure 'Protocol' is converted to numerical values for correlation calculation
    data['Protocol'] = data['Protocol'].astype('category').cat.codes

    # Correlation matrix
    plt.figure(figsize=(10, 8))
    corr_matrix = data[features + ['bad_packet']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    # Load data and analyze
    data = load_data()
    X, y = preprocess_data(data)
    analyze_data(data)
