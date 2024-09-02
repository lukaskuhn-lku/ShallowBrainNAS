import glob
import os
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, ttest_1samp, sem
import warnings


from scipy.stats import ttest_1samp
import pandas as pd
import numpy as np

def aggregate_df_by_layer(df):
    """
    Aggregates a single DataFrame by layer, averaging R values and computing combined significance,
    ensuring scalar values for each column.
    """
    aggregated_data = []

    for layer, group in df.groupby('Layer'):
        mean_r = group['R'].mean()
        t_stat, significance = ttest_1samp(group['R'], 0)
        
        # Assuming all rows within a single DataFrame have the same ROI
        common_roi_name = find_common_roi_name(group['ROI'].tolist())

        layer_data = {
            'ROI': common_roi_name,
            'Layer': layer,
            'Model': group['Model'].iloc[0],
            'R': mean_r,  # Use scalar value
            '%R2': mean_r ** 2,  # Use scalar value for %R2, computed from mean_r
            'Significance': significance,  # Use scalar value
            'SEM': group['R'].sem(),  # Use scalar value for SEM, if needed
            'LNC': np.nan,  # Placeholder for LNC, adjust as needed
            'UNC': np.nan  # Placeholder for UNC, adjust as needed
        }

        aggregated_data.append(layer_data)

    return pd.DataFrame(aggregated_data)


def find_common_roi_name(names):
    """
    Identifies the common ROI name within a single DataFrame.
    """
    if len(names) == 1:
        return names[0]  # Directly return the name if there's only one

    split_names = [name.split('_') for name in names]
    common_parts = set(split_names[0]).intersection(*split_names[1:])
    common_roi_name = '_'.join(common_parts)
    return common_roi_name

def aggregate_layers(dataframes):
    """
    Processes each DataFrame independently to aggregate by layer, then combines the results.
    Each DataFrame represents its own ROI, maintaining a single aggregated value per layer.
    """
    # Ensure dataframes is a list
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]

    aggregated_dfs = []

    for df in dataframes:
        aggregated_df = aggregate_df_by_layer(df)
        aggregated_dfs.append(aggregated_df)

    # Combine aggregated results from all DataFrames
    final_df = pd.concat(aggregated_dfs, ignore_index=True)
    
    return final_df


def get_layers_ncondns(feat_path):
    """
    Extracts information about the number of layers, the list of layer names, and the number of conditions (images)
    from the npz files in the specified feature path.

    Parameters:
    - feat_path (str): Path to the directory containing npz files with model features.

    Returns:
    - num_layers (int): The number of layers found in the npz files.
    - layer_list (list of str): A list containing the names of the layers.
    - num_conds (int): The number of conditions (images) based on the number of npz files in the directory.
    """
    
    # Find all npz files in the specified directory
    activations = glob.glob(feat_path + "/*.npz")
    
    # Count the number of npz files as the number of conditions (images)
    num_condns = len(activations)
    
    # Load the first npz file to extract layer information
    feat = np.load(activations[0], allow_pickle=True)

    num_layers = 0
    layer_list = []

    # Iterate through the keys in the npz file, ignoring metadata keys
    for key in feat:
        if "__" in key:  # key: __header__, __version__, __globals__
            continue
        else:
            num_layers += 1
            layer_list.append(key)  # collect all layer names ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

    return num_layers, layer_list, num_condns

def encode_layer(layer_id, n_components, batch_size, trn_Idx, tst_Idx, feat_path):
    """
    Encodes the layer activations using IncrementalPCA, for both training and test sets.

    Parameters:
    - layer_id (str): The layer name whose activations are to be encoded.
    - n_components (int): Number of components for PCA.
    - batch_size (int): Batch size for IncrementalPCA.
    - trn_Idx (list of int): Indices of the training set files.
    - tst_Idx (list of int): Indices of the test set files.
    - feat_path (str): Path to the directory containing npz files with model features.

    Returns:
    - pca_trn (numpy.ndarray): PCA-encoded features of the training set.
    - pca_tst (numpy.ndarray): PCA-encoded features of the test set.
    """
    activations = []
    feat_files = glob.glob(feat_path + '/*.npz')
    feat_files.sort()  # Ensure consistent order

    # Load a sample feature to check its dimensions after processing
    sample_feat = np.load(feat_files[0], allow_pickle=True)[layer_id]
    processed_sample_feat = np.mean(sample_feat, axis=1).flatten()

    # Determine whether to use PCA based on the dimensionality of the processed features
    #use_pca = processed_sample_feat.ndim > 1 or (processed_sample_feat.ndim == 1 and processed_sample_feat.shape[0] > 1)
    use_pca = False

    if use_pca:
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        for jj,ii in enumerate(trn_Idx):  # for each datafile for the current layer
            feat = np.load(feat_files[ii], allow_pickle=True)  # get activations of the current layer
            activations.append(np.mean(feat[layer_id], axis=1).flatten())
        
            # Partially fit the PCA model in batches
            if ((jj + 1) % batch_size) == 0 or (jj + 1) == len(trn_Idx):
                pca.partial_fit(np.stack(activations[-batch_size:],axis=0))
                
        # Transform the training set using the trained PCA model
        pca_trn = pca.transform(np.stack(activations,axis=0))
        
        # Repeat the process for the test set
        activations = []
        for ii in tst_Idx:  # for each datafile for the current layer
            feat = np.load(feat_files[ii], allow_pickle=True)  # get activations of the current layer
            activations.append(np.mean(feat[layer_id], axis=1).flatten())
        pca_tst = pca.transform(np.stack(activations,axis=0))
    else:
        #print("use no pca")
        # Directly use the activations without PCA transformation and ensure they are reshaped to 2D arrays
        
        #pca_trn = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in trn_Idx]).reshape(-1, 1)
        #pca_tst = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in tst_Idx]).reshape(-1, 1)

        trn = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in trn_Idx])
        tst = np.array([np.mean(np.load(feat_files[ii], allow_pickle=True)[layer_id], axis=1).flatten() for ii in tst_Idx])
        
        #print(trn.shape)
        #print(tst.shape)
    return trn, tst


def train_Ridgeregression_per_ROI(trn_x,tst_x,trn_y,tst_y,alpha):
    """
    Train a linear regression model for each ROI and compute correlation coefficients.

    Args:
        trn_x (numpy.ndarray): PCA-transformed training set activations.
        tst_x (numpy.ndarray): PCA-transformed test set activations.
        trn_y (numpy.ndarray): fMRI training set data.
        tst_y (numpy.ndarray): fMRI test set data.

    Returns:
        correlation_lst (numpy.ndarray): List of correlation coefficients for each ROI.
    """
    #reg = LinearRegression().fit(trn_x, trn_y)
    reg = Ridge(alpha=alpha)
    
    #print('for training:', trn_x.shape)
    #print('for training:', trn_y.shape)
    
    reg.fit(trn_x, trn_y)
    y_prd = reg.predict(tst_x)
    correlation_lst = np.zeros(y_prd.shape[1])
    for v in range(y_prd.shape[1]):
        correlation_lst[v] = pearsonr(y_prd[:,v], tst_y[:,v])[0]
    return correlation_lst


def Ridge_Encoding(feat_path, roi_path, model_name, alpha, trn_tst_split=0.8, n_folds=3, n_components=100, batch_size=100, just_corr=True, return_correlations = False,random_state=14, save_path="Linear_Encoding_Results"):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str or list): Path to the directory containing .npy fMRI data files for multiple ROIs.
        
            If we have a list of folders, each folders content will be summarized into one value. This is important if one folder contains data for the same ROI for different subjects
            
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        just_corr (bool): If True, only correlation values are considered in analysis (currently not used in function body).
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """
    
    # Check if its a list
    roi_paths = roi_path if isinstance(roi_path, list) else [roi_path]
    
    list_dataframes = []
    
    # Iterate through all folder paths
    for roi_path in roi_paths:
        result_dataframe = _ridge_encoding(feat_path, 
                                            roi_path, 
                                            model_name,
                                            alpha,
                                            trn_tst_split=trn_tst_split, 
                                            n_folds=n_folds, 
                                            n_components=n_components, 
                                            batch_size=batch_size,
                                            just_corr=just_corr, 
                                            return_correlations=return_correlations,
                                            random_state=random_state)
        

        # Collect dataframes in list
        list_dataframes.append(result_dataframe[0])
    
    # If just one dataframe, return it as it is
    if len(list_dataframes) == 1:
        final_df = list_dataframes[0]
    else:
        final_df = aggregate_layers(list_dataframes)
        
    # Create the output folder if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    csv_file_path = f"{save_path}/{model_name}.csv"
    final_df.to_csv(csv_file_path, index=False)
    
    return final_df
        
        
        
        
def ridge_encoding(*args, **kwargs):
    warnings.warn(
        "The 'linear_encoding' function is deprecated and has been replaced by 'Linear_Encoding'. "
        "Please update your code to use the new function name, as this alias will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    return Ridge_Encoding(*args, **kwargs)   
        
    

def _ridge_encoding(feat_path, roi_path, model_name, alpha, trn_tst_split=0.8, n_folds=3, n_components=100, batch_size=100, just_corr=True, return_correlations = False,random_state=14):
    """
    Perform linear encoding analysis to relate model activations to fMRI data across multiple folds.

    Args:
        feat_path (str): Path to the directory containing model activation .npz files for multiple layers.
        roi_path (str): Path to the directory containing .npy fMRI data files for multiple ROIs.
        model_name (str): Name of the model being analyzed (used for labeling in the output).
        trn_tst_split (float): Proportion of data to use for training (rest is used for testing).
        n_folds (int): Number of folds to split the data for cross-validation.
        n_components (int): Number of principal components to retain in PCA.
        batch_size (int): Batch size for Incremental PCA.
        just_corr (bool): If True, only correlation values are considered in analysis (currently not used in function body).
        return_correlations (bool): If True, return correlation values for each ROI and layer.
        random_state (int): Seed for random operations to ensure reproducibility.

    Returns:
        all_rois_df (pd.DataFrame): DataFrame summarizing the analysis results including correlations and statistical significance.
        corr_dict (dict): Dictionary containing correlation values for each layer and ROI (only if return_correlations is True).
    """
    
    # Initialize dictionaries to store results
    fold_dict = {}  # To store fold-wise results
    corr_dict = {}  # To store correlations if requested
    
    # Check if roi_path is a list, if not, make it a list
    roi_paths = roi_path if isinstance(roi_path, list) else [roi_path]
    
    # Load feature files and get layer information
    feat_files = glob.glob(feat_path+'/*.npz')
    num_layers, layer_list, num_condns = get_layers_ncondns(feat_path)
    
    # Create a tqdm object with an initial description
    #pbar = tqdm(range(n_folds), desc="Initializing folds")
    
    # Loop over each fold for cross-validation
    for fold_ii in range(n_folds):
        #pbar.set_description(f"Processing fold {fold_ii + 1}/{n_folds}")
        
        # Set random seeds for reproducibility
        np.random.seed(fold_ii+random_state)
        random.seed(fold_ii+random_state)
        
        # Split the data indices into training and testing sets
        trn_Idx,tst_Idx = train_test_split(range(len(feat_files)),test_size=(1-trn_tst_split),train_size=trn_tst_split,random_state=fold_ii+random_state)
        
        # Process each layer of model activations
        for layer_id in layer_list:
            if layer_id not in fold_dict.keys():
                fold_dict[layer_id] = {}
                corr_dict[layer_id] = {}
            
            # Encode the current layer using PCA and split into training and testing sets
            pca_trn,pca_tst = encode_layer(layer_id, n_components, batch_size, trn_Idx, tst_Idx, feat_path)

            for roi_path in roi_paths:
                roi_files = []
            
                 # Check if the roi_path is a file or a directory
                if os.path.isfile(roi_path) and roi_path.endswith('.npy'):
                    # If it's a file, directly use it
                    roi_files.append(roi_path)
                elif os.path.isdir(roi_path):
                    # If it's a directory, list all .npy files within it
                    roi_files.extend(glob.glob(os.path.join(roi_path, '*.npy')))
                else:
                    print(f"Invalid ROI path: {roi_path}")
                    continue  # Skip this roi_path if it's neither a valid file nor a directory

                # Process each ROI's fMRI data
                if not roi_files:
                    print(f"No roi_files found in {roi_path}")
                    continue  # Skip to the next roi_path if no ROI files were found

                for roi_file in roi_files:
                    roi_name = os.path.basename(roi_file)[:-4]
                    if roi_name not in fold_dict[layer_id].keys():
                        fold_dict[layer_id][roi_name] = []
                        corr_dict[layer_id][roi_name] = []
                        
                    # Load fMRI data for the current ROI and split into training and testing sets
                    fmri_data = np.load(os.path.join(roi_file))
                    fmri_trn,fmri_tst = fmri_data[trn_Idx],fmri_data[tst_Idx]
                    
                    # Train a linear regression model and compute correlations for the current ROI
                    r_lst = train_Ridgeregression_per_ROI(pca_trn,pca_tst,fmri_trn,fmri_tst,alpha)
                    r = np.mean(r_lst) # Mean of all train test splits
                    
                    # Store correlation results
                    if return_correlations:                   
                        corr_dict[layer_id][roi_name].append(r_lst)
                        if fold_ii == n_folds-1:
                            corr_dict[layer_id][roi_name] = np.mean(np.array(corr_dict[layer_id][roi_name], dtype=np.float16),axis=0)
                    fold_dict[layer_id][roi_name].append(r)
                
    # Compile all results into a DataFrame for easy analysis
    # Define the column types explicitly
    column_types = {
        'ROI': str,
        'Layer': str,
        'Model': str,
        'R': float,
        '%R2': float,
        'Significance': float,
        'SEM': float,
        'LNC': float,
        'UNC': float
    }

    # Initialize an empty list to store individual DataFrames
    all_rois_list = []

    for layer_id, layer_dict in fold_dict.items():
        for roi_name, r_lst in layer_dict.items():
            # Compute statistical significance of the correlations
            r_lst_array = np.array(r_lst)  # Convert the list to a NumPy array
            significance = ttest_1samp(r_lst_array, 0)[1]
            R = np.mean(r_lst_array)
            
            output_dict = {
                "ROI": roi_name,
                "Layer": layer_id,
                "Model": model_name,
                "R": R,
                "%R2": R ** 2,
                "Significance": significance,
                "SEM": sem(r_lst_array),
                "LNC": 0,  # or np.nan if you prefer
                "UNC": 0   # or np.nan if you prefer
            }
            
            # Create a DataFrame for this layer/ROI combination
            layer_df = pd.DataFrame([output_dict])
            all_rois_list.append(layer_df)

    # Concatenate all DataFrames at once
    all_rois_df = pd.concat(all_rois_list, ignore_index=True)

    # Ensure all columns have the correct data type
    all_rois_df = all_rois_df.astype(column_types)
            
    if return_correlations:
        return all_rois_df,corr_dict
    
    return all_rois_df
