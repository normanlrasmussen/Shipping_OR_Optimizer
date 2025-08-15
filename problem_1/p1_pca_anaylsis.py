import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import ast

# data = [w, o, c, s]
# w - washers
# o - oven
# c - couches
# s - sofas

def generate_plot(output_file):
    """
    Generate a 2D PCA visualization of the truck packing problem data.
    
    This function performs Principal Component Analysis on the truck packing data,
    which contains combinations of washers, ovens, couches, and sofas, along with
    the minimum number of trucks needed to pack each combination. The PCA reduces
    the 4-dimensional feature space to 2 dimensions for visualization.
    
    Args:
        output_file (str): Path where the plot image will be saved.
    
    Returns:
        None: The function saves a plot to the specified file and displays it.
    
    Features:
        - Parses the 'orders' column from CSV to extract individual item counts
        - Standardizes the data using StandardScaler
        - Performs PCA with 2 components
        - Creates a scatter plot colored by the number of trucks needed
        - Displays eigenvectors in the bottom right corner
        - Shows total sample count in the bottom left corner
        - Includes a legend for truck counts
    """
    df = pd.read_csv("problem_1/problem_1_data.csv")
    df = df.dropna()
    df = df[df['min_num_of_trucks'] != 0]
    
    # Parse the 'orders' column to extract individual values
    # Convert string representations of lists to actual lists
    orders_lists = [ast.literal_eval(order_str) for order_str in df['orders']]
    
    # Extract individual values into separate columns
    df['washers'] = [order[0] for order in orders_lists]
    df['ovens'] = [order[1] for order in orders_lists]
    df['couches'] = [order[2] for order in orders_lists]
    df['sofas'] = [order[3] for order in orders_lists]
    
    # Prepare data for PCA
    # Select the feature columns (washers, ovens, couches, sofas)
    feature_columns = ['washers', 'ovens', 'couches', 'sofas']
    X = df[feature_columns].values  # Convert to numpy array
    
    # You can also include the target variable (min_num_of_trucks) for analysis
    y = df['min_num_of_trucks'].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Prepare eigenvector text for bottom right corner
    eigenvector_text = "Eigenvectors (PC1, PC2):\n"
    for j, feature in enumerate(feature_columns):
        eigenvector_text += f"{feature}: [{pca.components_[0, j]:.2f}, {pca.components_[1, j]:.2f}]\n"
    
    # Create a color map for the number of trucks
    unique_trucks = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_trucks)))
    
    # Plot each point with color based on number of trucks
    for i in range(len(X_pca)):
        truck_idx = np.where(unique_trucks == y[i])[0][0]
        plt.plot(X_pca[i, 0], X_pca[i, 1], 'o', color=colors[truck_idx], alpha=0.5)
    
    # Add legend
    for i, truck_count in enumerate(unique_trucks):
        plt.plot([], [], 'o', color=colors[i], label=f'{truck_count} trucks')
    plt.legend()
    
    plt.title("PCA of Problem 1 Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    
    # Add text in bottom left corner
    plt.text(0.02, 0.02, f'Total samples: {len(X_pca)}', 
             transform=plt.gca().transAxes, 
             fontsize=10, 
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add eigenvector text in bottom right corner
    plt.text(0.98, 0.02, eigenvector_text, 
             transform=plt.gca().transAxes, 
             fontsize=8, 
             verticalalignment='bottom',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()

def generate_plot_3d():
    """
    Generate a 3D PCA visualization of the truck packing problem data.
    
    This function performs Principal Component Analysis on the truck packing data,
    reducing the 4-dimensional feature space to 3 dimensions for 3D visualization.
    This provides an additional perspective on the data structure and clustering
    patterns that may not be visible in 2D.
    
    Args:
        None: No parameters required.
    
    Returns:
        None: The function displays a 3D interactive plot.
    
    Features:
        - Parses the 'orders' column from CSV to extract individual item counts
        - Standardizes the data using StandardScaler
        - Performs PCA with 3 components
        - Creates a 3D scatter plot colored by the number of trucks needed
        - Includes a legend for truck counts
        - Provides interactive 3D visualization with rotation and zoom capabilities
    """
    df = pd.read_csv("problem_1/problem_1_data.csv")
    df = df.dropna()
    df = df[df['min_num_of_trucks'] != 0]
    
    # Parse the 'orders' column to extract individual values
    # Convert string representations of lists to actual lists
    orders_lists = [ast.literal_eval(order_str) for order_str in df['orders']]
    
    # Extract individual values into separate columns
    df['washers'] = [order[0] for order in orders_lists]
    df['ovens'] = [order[1] for order in orders_lists]
    df['couches'] = [order[2] for order in orders_lists]
    df['sofas'] = [order[3] for order in orders_lists]
    
    # Prepare data for PCA
    # Select the feature columns (washers, ovens, couches, sofas)
    feature_columns = ['washers', 'ovens', 'couches', 'sofas']
    X = df[feature_columns].values  # Convert to numpy array
    
    # You can also include the target variable (min_num_of_trucks) for analysis
    y = df['min_num_of_trucks'].values
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create a color map for the number of trucks
    unique_trucks = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_trucks)))
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each point with color based on number of trucks
    for i in range(len(X_pca)):
        truck_idx = np.where(unique_trucks == y[i])[0][0]
        ax.scatter(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], color=colors[truck_idx], alpha=0.5)
    
    # Add legend
    for i, truck_count in enumerate(unique_trucks):
        ax.scatter([], [], [], color=colors[i], label=f'{truck_count} trucks')
    plt.legend()
    
    plt.title("PCA of Problem 1 Data")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_zlabel("Principal Component 3")

    plt.show()


if __name__ == "__main__":
    generate_plot("problem_1/pca_p1.png")
    generate_plot_3d()