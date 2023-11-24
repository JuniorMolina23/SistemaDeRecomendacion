import pandas as pd
import numpy as np

def load_and_consolidate_data(input_file, num_rows=100000):
    # Load MovieLens data using Pandas
    ratings = pd.read_csv(input_file, sep='\t', header=None,
                          names=['userId', 'movieId', 'rating', 'rating_timestamp'],
                          nrows=num_rows)  # Lee solo las primeras 'num_rows' filas

    # Convert to NumPy array
    np_data = np.array(ratings[['userId', 'movieId', 'rating']])

    # Convert NumPy array to Pandas DataFrame
    df = pd.DataFrame(np_data, columns=['userId', 'movieId', 'rating'])

    # Consolidate dataimport pandas as pd
import numpy as np

def load_and_consolidate_data(input_file, num_rows=100000):
    # Load MovieLens data using Pandas
    ratings = pd.read_csv(input_file, sep='\t', header=None,
                          names=['userId', 'movieId', 'rating', 'rating_timestamp'],
                          nrows=num_rows)  # Lee solo las primeras 'num_rows' filas

    # Convert to NumPy array
    np_data = np.array(ratings[['userId', 'movieId', 'rating']])

    # Convert NumPy array to Pandas DataFrame
    df = pd.DataFrame(np_data, columns=['userId', 'movieId', 'rating'])

    # Consolidate data
    consolidated_df = df.groupby(['userId', 'movieId'])['rating'].mean().unstack()

    return consolidated_df

# Load and consolidate data once

def limpia(np1, np2):
    mask = ~np.isnan(np2)
    np1 = np1[mask]
    np2 = np2[mask]

    np1, np2 = np2, np1

    mask = ~np.isnan(np2)
    np1 = np1[mask]
    np2 = np2[mask]

    np1, np2 = np2, np1

    return pd.DataFrame({'A': np1, 'B': np2})

def computeManhattanDistance(ax, bx):
    return np.sum(np.abs(ax - bx))

def computeEuclideanDistance(ax, bx):
    return np.sqrt(np.sum((ax - bx)**2))

def computeNearestNeighbor(username, users_df, distance):
    user_data = np.array(users_df.loc[username])
    distances = np.empty((users_df.shape[0],), dtype=float)

    for i, (index, row) in enumerate(users_df.iterrows()):
        if index != username:
            ax = np.array(row)
            bx = np.array(user_data)
            temp = limpia(ax, bx)
            ax = np.array(temp["A"].tolist())
            bx = np.array(temp["B"].tolist())
            distances[i] = distance(ax, bx)

    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]

    return list(zip(sorted_distances, users_df.index[sorted_indices]))

def neighbors(user, distance, consolidated_df):
    username = user
    nearest_neighbors = computeNearestNeighbor(username, consolidated_df, distance)
    primeros_tres_neighbors = nearest_neighbors[:3]
    return primeros_tres_neighbors

def neighbors_options_distances (user, op, consolidated_df):
    if op == 1:
        return str(neighbors(user, computeManhattanDistance, consolidated_df)), "Manhattan"
    elif op == 2:
        return str(neighbors(user, computeEuclideanDistance, consolidated_df)), "Euclidiana"
    else:
        return 'no encontrada'
    
    consolidated_df = df.groupby(['userId', 'movieId'])['rating'].mean().unstack()

    return consolidated_df

# Load and consolidate data once

def limpia(np1, np2):
    mask = ~np.isnan(np2)
    np1 = np1[mask]
    np2 = np2[mask]

    np1, np2 = np2, np1

    mask = ~np.isnan(np2)
    np1 = np1[mask]
    np2 = np2[mask]

    np1, np2 = np2, np1

    return pd.DataFrame({'A': np1, 'B': np2})

def computeManhattanDistance(ax, bx):
    return np.sum(np.abs(ax - bx))

def computeEuclideanDistance(ax, bx):
    return np.sqrt(np.sum((ax - bx)**2))

def computeNearestNeighbor(username, users_df, distance):
    user_data = np.array(users_df.loc[username])
    distances = np.empty((users_df.shape[0],), dtype=float)

    for i, (index, row) in enumerate(users_df.iterrows()):
        if index != username:
            ax = np.array(row)
            bx = np.array(user_data)
            temp = limpia(ax, bx)
            ax = np.array(temp["A"].tolist())
            bx = np.array(temp["B"].tolist())
            distances[i] = distance(ax, bx)

    sorted_indices = np.argsort(distances)
    sorted_distances = distances[sorted_indices]

    return list(zip(sorted_distances, users_df.index[sorted_indices]))

def neighbors(user, distance, consolidated_df):
    username = user
    nearest_neighbors = computeNearestNeighbor(username, consolidated_df, distance)
    primeros_tres_neighbors = nearest_neighbors[:3]
    return primeros_tres_neighbors

def neighbors_options_distances (user, op, consolidated_df):
    if op == 1:
        return str(neighbors(user, computeManhattanDistance, consolidated_df)), "Manhattan"
    elif op == 2:
        return str(neighbors(user, computeEuclideanDistance, consolidated_df)), "Euclidiana"
    else:
        return 'no encontrada'
    