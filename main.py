from src.utils import load_dataset, extract_year, filter_data, get_top_n_results, preprocess_text
from src.data_structures.kd_tree import KDTree
from src.data_structures.quad_tree import QuadTree
from src.data_structures.range_tree import RangeTree
from src.data_structures.r_tree import RTree
from src.lsh import *

import sys
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np

def test_kd_tree(data):
    # Build a k-d tree with relevant attributes
    points = data[['100g_USD', 'rating']].values.tolist()
    kd_tree = KDTree(points)

    print("k-d Tree built successfully!")

    # Insert a new point
    kd_tree.insert_point([7.5, 96])

    # Perform a range query: Price between 4-10 USD and Rating > 94
    query_range = [(4, 10), (94, 100)]
    results = kd_tree.search_range(query_range)

    print("Range Query Results:", results)

def test_quad_tree(data):
    points = data[['100g_USD', 'rating']].values.tolist()

    # Define the boundary of the root node (e.g., price and rating ranges)
    boundary = (0, 20, 80, 100)  # Price between 0-20 USD, Rating 80-100
    capacity = 4  # Max points per node before splitting
    quad_tree = QuadTree(boundary, capacity)

    # Insert points into the QuadTree
    for point in points:
        quad_tree.insert_point(point)

    # Perform a range query: Price between 4-10 USD and Rating > 94
    query_range = (4, 10, 94, 100)
    results = quad_tree.search_range(query_range)

    print("Range Query Results:", results)

def test_range_tree(data):
    points = data[['100g_USD', 'rating']].values.tolist()

    # Build the Range Tree
    range_tree = RangeTree(points)

    # Perform a range query: Price between 4-10 USD and Rating > 94
    query_range = [(4, 10), (94, 100)]
    results = range_tree.search_range(query_range)

    print("Range Query Results:", results)

def test_r_tree(data):
    points = data[['100g_USD', 'rating']].values.tolist()

    # Build the R-Tree
    r_tree = RTree(max_entries=4)
    for point in points:
        r_tree.insert(point)

    # Perform a range query: Price between 4-10 USD and Rating > 94
    query_box = (4, 94, 10, 100)
    results = r_tree.range_query(r_tree.root, query_box)

    print("Range Query Results:", results)

def test_query_processing(data):
    """
    Test the query processing pipeline with filtering and top-N selection.
    :param data: Pandas DataFrame.
    """
    # Extract the year from the review_date column
    data["review_date_year"] = data["review_date"].apply(extract_year)

    # Print the review_date and extracted review_date_year for verification
    #print("Preview of Year Extraction:")
    #print(data[["review_date", "review_date_year"]].head(10))  # Display the first 10 rows

    # Define query parameters
    query = {
        "year_range": (2019, 2021),
        "price_range": (4, 10),
        "min_rating": 85,
        "loc_country": "United States"
    }

    # Apply filters
    #commended out lines are for debugging
    #print("Query Parameters:")
    #print(query)
    #print("Dataset Preview:")
    #print(data.head())
    #print("Check for Missing Values:")
    #print(data.isnull().sum())
    filtered_data = filter_data(data, query)
    #print("Filtered Data:")
    #print(filtered_data)
    #print("i hope it worked")
    sys.stdout.reconfigure(encoding='utf-8')
    #print(data)

    #uncomment these for full functionality
    # Get top-3 results
    top_results = get_top_n_results(filtered_data, n=3)

    #print("Filtered Data:")
    #print(filtered_data)
    #print("\nTop-3 Results:")
    #print(top_results)
    
    return filtered_data

def test_data_structures(filtered_data):
    # Convert to points for the KDTree
    points = filtered_data[['100g_USD', 'rating']].values.tolist()

    # Test KDTree
    kd_tree = KDTree(points)
    print("KDTree built successfully. Testing range query...")
    query_range = [(4, 10), (94, 100)]
    kd_results = kd_tree.search_range(query_range)
    print("KDTree Results:", kd_results)

    # Similarly, you can test QuadTree, RangeTree, and RTree -not needed tho, we already have functions for them


########APO EDW KAI KATW FUNCTIONS GIA LSH###############################


def kd_tree_with_lsh(filtered_data, kd_query_range):
    """
    Perform KDTree range query and run LSH pipeline on the already filtered data.
    :param filtered_data: Pre-filtered Pandas DataFrame.
    :param kd_query_range: Range query for KDTree.
    """
    if filtered_data.empty:
        print("Filtered data is empty. Nothing to process.")
        return

    print("Filtered Data:")
    print(filtered_data)

    # Step 1: Build KDTree
    print("Building KDTree...")
    points = filtered_data[['100g_USD', 'rating']].values.tolist()
    kd_tree = KDTree(points)

    # Step 2: Perform KDTree range query
    print("Performing KDTree range query...")
    results = kd_tree.search_range(kd_query_range)
    print("KDTree Range Query Results:")
    print(results)

    if not results:
        print("No data found in the specified KDTree range.")
        return

    # Step 3: Subset the filtered data based on KDTree range query results
    print("Filtering data based on KDTree range query results...")
    import numpy as np
    filtered_points = filtered_data[['100g_USD', 'rating']].values
    final_filtered_data = filtered_data[
        np.any([np.isclose(filtered_points, result, atol=1e-2).all(axis=1) for result in results], axis=0)
    ]

    if final_filtered_data.empty:
        print("No data left after KDTree range query filtering.")
        return

    print("Final Filtered Data for LSH:")
    print(final_filtered_data)

    # Step 4: Run LSH pipeline on the final filtered data
    print("Running LSH pipeline on final filtered data...")
    test_lsh_pipeline(final_filtered_data)

    # Return the range query results and tree name for visualization
    return results, "k-d Tree"


def quad_tree_with_lsh(filtered_data, query_range):
    """
    Perform QuadTree range query and run LSH pipeline on the already filtered data.
    :param filtered_data: Pre-filtered Pandas DataFrame.
    :param query_range: Range query for the QuadTree (min_price, max_price, min_rating, max_rating).
    """
    if filtered_data.empty:
        print("Filtered data is empty. Nothing to process.")
        return

    print("Filtered Data:")
    print(filtered_data)

    # Step 1: Build QuadTree
    print("Building QuadTree...")
    points = filtered_data[['100g_USD', 'rating']].values.tolist()
    boundary = (0, 20, 80, 100)  # Define the boundary (can be adjusted)
    capacity = 4  # Max points per node before splitting
    quad_tree = QuadTree(boundary, capacity)

    for point in points:
        quad_tree.insert_point(point)

    # Step 2: Perform QuadTree range query
    print("Performing QuadTree range query...")
    results = quad_tree.search_range(query_range)
    print("QuadTree Range Query Results:")
    print(results)

    if not results:
        print("No data found in the specified QuadTree range.")
        return

    # Step 3: Subset the filtered data based on QuadTree range query results
    print("Filtering data based on QuadTree range query results...")
    import numpy as np
    filtered_points = filtered_data[['100g_USD', 'rating']].values
    final_filtered_data = filtered_data[
        np.any([np.isclose(filtered_points, result, atol=1e-2).all(axis=1) for result in results], axis=0)
    ]

    if final_filtered_data.empty:
        print("No data left after QuadTree range query filtering.")
        return

    print("Final Filtered Data for LSH:")
    print(final_filtered_data)

    # Step 4: Run LSH pipeline on the final filtered data
    print("Running LSH pipeline on final filtered data...")
    test_lsh_pipeline(final_filtered_data)

    # Return the range query results and tree name for visualization
    return results, "QuadTree"


def range_tree_with_lsh(filtered_data, query_range):
    """
    Perform Range Tree query and run LSH pipeline on the already filtered data.
    :param filtered_data: Pre-filtered Pandas DataFrame.
    :param query_range: Range query for the Range Tree (e.g., [(x_min, x_max), (y_min, y_max)]).
    """
    if filtered_data.empty:
        print("Filtered data is empty. Nothing to process.")
        return

    print("Filtered Data:")
    print(filtered_data)

    # Step 1: Build Range Tree
    print("Building Range Tree...")
    points = filtered_data[['100g_USD', 'rating']].values.tolist()
    range_tree = RangeTree(points)

    # Step 2: Perform Range Tree query
    print("Performing Range Tree query...")
    if not isinstance(query_range, list) or len(query_range) != 2:
        raise ValueError("query_range must be a list of two tuples: [(x_min, x_max), (y_min, y_max)]")

    results = range_tree.search_range(query_range)
    print("Range Tree Query Results:")
    print(results)

    if not results:
        print("No data found in the specified Range Tree range.")
        return

    # Step 3: Subset the filtered data based on Range Tree query results
    print("Filtering data based on Range Tree query results...")
    import numpy as np
    filtered_points = filtered_data[['100g_USD', 'rating']].values
    final_filtered_data = filtered_data[
        np.any([np.isclose(filtered_points, result, atol=1e-2).all(axis=1) for result in results], axis=0)
    ]

    if final_filtered_data.empty:
        print("No data left after Range Tree query filtering.")
        return

    print("Final Filtered Data for LSH:")
    print(final_filtered_data)

    # Step 4: Run LSH pipeline on the final filtered data
    print("Running LSH pipeline on final filtered data...")
    test_lsh_pipeline(final_filtered_data)

    # Return the range query results and tree name for visualization
    return results, "RangeTree"


def r_tree_with_lsh(filtered_data, query_box):
    """
    Perform R-Tree range query and run LSH pipeline on the already filtered data.
    :param filtered_data: Pre-filtered Pandas DataFrame.
    :param query_box: Range query for the R-Tree (x_min, y_min, x_max, y_max).
    """
    if filtered_data.empty:
        print("Filtered data is empty. Nothing to process.")
        return

    print("Filtered Data:")
    print(filtered_data)

    # Step 1: Build R-Tree
    print("Building R-Tree...")
    points = filtered_data[['100g_USD', 'rating']].values.tolist()
    r_tree = RTree(max_entries=4)

    for point in points:
        r_tree.insert(point)

    # Step 2: Perform R-Tree range query
    print("Performing R-Tree range query...")
    if not isinstance(query_box, tuple) or len(query_box) != 4:
        raise ValueError("query_box must be a tuple with 4 values: (x_min, y_min, x_max, y_max)")

    results = r_tree.range_query(r_tree.root, query_box)
    print("R-Tree Range Query Results:")
    print(results)

    if not results:
        print("No data found in the specified R-Tree range.")
        return

    # Step 3: Subset the filtered data based on R-Tree range query results
    print("Filtering data based on R-Tree range query results...")
    import numpy as np
    filtered_points = filtered_data[['100g_USD', 'rating']].values
    final_filtered_data = filtered_data[
        np.any([np.isclose(filtered_points, result, atol=1e-2).all(axis=1) for result in results], axis=0)
    ]

    # Explicitly filter results to ensure they are within the query range
    x_min, y_min, x_max, y_max = query_box
    results = [
        point for point in results
        if x_min <= point[0] <= x_max and y_min <= point[1] <= y_max
    ]

    if final_filtered_data.empty:
        print("No data left after R-Tree range query filtering.")
        return

    print("Final Filtered Data for LSH:")
    print(final_filtered_data)

    # Step 4: Run LSH pipeline on the final filtered data
    print("Running LSH pipeline on final filtered data...")
    test_lsh_pipeline(final_filtered_data)

    # Return the range query results and tree name for visualization
    return results, "R-Tree"

####COMPARE DATA STRUCTURES#######################################################

def compare_data_structures(data, query_params):
    """
    Automatically compare k-d tree, QuadTree, RangeTree, and R-Tree in terms of execution time.
    :param data: The dataset to test.
    :param query_params: Query parameters (e.g., range query and LSH configuration).
    """
    # Dictionary to store performance results
    performance = {
        'k-d Tree': {'build_time': [], 'query_time': []},
        'QuadTree': {'build_time': [], 'query_time': []},
        'RangeTree': {'build_time': [], 'query_time': []},
        'R-Tree': {'build_time': [], 'query_time': []}
    }

    # Function to preprocess query ranges
    def preprocess_query(query_params, tree_type):
        if tree_type in ['k-d Tree', 'RangeTree']:
            # Convert (x1, x2, y1, y2) -> [(x1, x2), (y1, y2)]
            return [(query_params[0], query_params[1]), (query_params[2], query_params[3])]
        elif tree_type in ['R-Tree']:
            # Convert (x1, x2, y1, y2) -> (x1, y1, x2, y2)
            return (query_params[0], query_params[2], query_params[1], query_params[3])
        elif tree_type in ['QuadTree', 'R-Tree']:
            # Use (x1, x2, y1, y2) format as is
            return query_params

    # Testing each data structure
    for name, tree_function in {
        'k-d Tree': kd_tree_with_lsh,
        'QuadTree': quad_tree_with_lsh,
        'RangeTree': range_tree_with_lsh,
        'R-Tree': r_tree_with_lsh
    }.items():
        print(f"Testing {name}...")

        # Preprocess the query params for the specific tree
        preprocessed_query = preprocess_query(query_params, name)

        # Measure build time
        start_time = time.time()
        filtered_data = tree_function(data, preprocessed_query)  # Build and run the tree
        build_time = time.time() - start_time
        query_time = time.time() - start_time


        # Measure query time (LSH pipeline inside tree function)
        #start_time = time.time()
        #results = tree_function(data, preprocessed_query)
        #query_time = time.time() - start_time

        # Store results
        performance[name]['build_time'].append(build_time)
        performance[name]['query_time'].append(query_time)

        # Visualize results after each tree processes the query
        results, tree_name = tree_function(data, preprocessed_query)
        visualize_range_results(data, query_params, results, tree_name)


    # Generate comparison graph
    plot_performance(performance)

##########VISUALIZATION & PLOTS###############################################

import matplotlib.pyplot as plt

def visualize_range_results(filtered_data, query_range, results, tree_name):
    """
    Visualize the range query results on a scatter plot.
    :param filtered_data: Entire dataset used for range queries (Pandas DataFrame).
    :param query_range: Range query used for filtering.
    :param results: Points returned by the range query.
    :param tree_name: Name of the tree (for labeling purposes).
    """
    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot all data points with light opacity
    plt.scatter(
        filtered_data['100g_USD'], 
        filtered_data['rating'], 
        color='#c586c0', 
        label='All Data Points', 
        alpha=0.2  # Low opacity to make them less dominant
    )

    # Highlight range query results
    if results:
        results_x = [point[0] for point in results]
        results_y = [point[1] for point in results]
        plt.scatter(
            results_x, 
            results_y, 
            color='#6666ea', 
            label=f'{tree_name} Range Query Results', 
            edgecolor='black', 
            s=60  # Larger size to make them stand out more
        )

    # Visualize query range (as a rectangle)
    x_min, x_max, y_min, y_max = query_range if len(query_range) == 4 else (
        query_range[0][0], query_range[0][1], query_range[1][0], query_range[1][1]
    )
    plt.gca().add_patch(
        plt.Rectangle(
            (x_min, y_min), 
            x_max - x_min, 
            y_max - y_min, 
            color='blue', 
            alpha=0.2, 
            label='Query Range'
        )
    )

    # Optional: Add a zoom-in view
    plt.xlim([x_min - 1, x_max + 1])  # Zooming a little around the query range
    plt.ylim([y_min - 1, y_max + 1])

    # Labels and legend
    plt.title(f"{tree_name} Range Query Visualization")
    plt.xlabel("100g Price (USD)")
    plt.ylabel("Rating")
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()


def plot_performance(performance):
    """
    Plot performance metrics for all data structures.
    :param performance: Dictionary containing build and query times.
    """
    structures = list(performance.keys())
    build_times = [performance[s]['build_time'][0] for s in structures]
    query_times = [performance[s]['query_time'][0] for s in structures]

    x = range(len(structures))  # X-axis positions

    # Plot build and query times side by side
    plt.figure(figsize=(10, 6))
    plt.bar(x, build_times, width=0.4, label='Build Time', align='center', color='#6666ea')
    plt.bar(x, query_times, width=0.4, label='Query Time', align='edge', color='#df5320')

    # Add labels, title, and legend
    plt.xticks(x, structures)
    plt.xlabel('Data Structures')
    plt.ylabel('Time (seconds)')
    plt.title('Performance Comparison of Data Structures')
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    filepath = "data/simplified_coffee.csv"
    coffee_data = load_dataset(filepath)

    # Preprocess the data #DEN DOULEVEI LOGW PERIERGOU FORMAT STO CSV
    #coffee_data = preprocess_data(coffee_data)

    # Inspect preprocessed data
    #print(coffee_data.head())

    filtered_data = test_query_processing(coffee_data)

    #uncomment to test:

    # Test k-d tree
    #test_kd_tree(filtered_data)

    # Test quad tree
    #test_quad_tree(filtered_data)

    # Test range tree
    #test_range_tree(filtered_data)

    # Test r tree
    #test_r_tree(filtered_data)
    

    #test_data_structures(filtered_data)

    ##########TESTING LSH##########

    # Define the query range for the KDTree
    kd_query_range = [(4, 6), (90, 95)]  # Price and Rating 
    # Test KDTree with LSH pipeline
    #kd_tree_with_lsh(filtered_data, kd_query_range)


    # Define the query range for the QuadTree
    quad_query_range = (4, 6, 90, 95)  # Price and Rating  
    # Test QuadTree with LSH pipeline
    #quad_tree_with_lsh(filtered_data, quad_query_range)
    
    # Define the query range for the Range Tree
    range_query_range = [(4, 6), (90, 95)]  # Price and Rating
    # Test Range Tree with LSH pipeline
    #range_tree_with_lsh(filtered_data, range_query_range)
    
    # Define the query range for the R-Tree
    r_query_range = (4, 90, 6, 95)  # Price and Rating
    # Test R-Tree with LSH pipeline
    #r_tree_with_lsh(filtered_data, r_query_range)


    # Define a query range
    query_params = (4, 6, 90, 95)  

    # Run comparison
    #compare_data_structures(filtered_data, query_params)

    
    
if __name__ == "__main__":
    main()
