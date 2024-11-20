from itertools import product, chain, combinations
from collections import Counter

import numpy as np
import pandas as pd

from .Container import Container


def convert(
    data_frame,
    margin_threshold=1,
    product_label="name",
    count_label="count",
    ae_label="AE",
    count_unique_ids=False,
    id_label="id",
    opt=True
):
    """
    Convert a Pandas dataframe object into a container class for use
    with the disproportionality analyses. Column names in the DataFrame
    must include or be specified in the arguments:
        "name" -- A brand/generic name for the product. This module
                    expects that you have already cleaned the data
                    so there is only one name associated with a class.
        "AE" -- The adverse event(s) associated with a drug/device.
        "count" -- The number of AEs associated with that drug/device
                    and AE. You can input a sheet with single counts
                    (i.e. duplicate rows) or pre-aggregated counts

    Arguments:
        data_frame (Pandas DataFrame): The Pandas DataFrame object

        margin_threshold (int): The threshold for counts. Lower numbers will
                             be removed from consideration (This doesn't seem to work)

        count_unique_ids (bool): Essentially whether we count each drug/adverse event
                                 once per id or not
                                 Set True to get processRaw to behave the same as OpenEBGM

        product_label(str): Name of the column containing the products 

        count_label(str): Name of the column containing the counts 

        ae_label(str): Name of the column containing the adverse events 

        id_label(str): Name of the column that contains the id information

        opt (bool): Whether to use the optimised (vectorised) counting functions

    Returns:
        RES (DataStorage object): A container object that holds the necessary
                                    components for DA.

    """
    col_list = data_frame.columns

    if ((id_label not in col_list) and (count_unique_ids==True)):
        raise ValueError(f"Column '{id_label}' does not exist in the DataFrame")
    if product_label not in col_list:
        raise ValueError(f"Column '{product_label}' does not exist in the DataFrame")
    if ae_label not in col_list:
        raise ValueError(f"Column '{ae_label}' does not exist in the DataFrame")
    if count_label not in col_list:
        raise ValueError(f"Column '{count_label}' does not exist in the DataFrame")


    data_cont = compute_contingency(
        data_frame, product_label, count_label, ae_label, margin_threshold
    )


    if (not count_unique_ids):
        # Compute the flattened table from the contingency table 

        col_sums = np.sum(data_cont, axis=0)
        row_sums = np.sum(data_cont, axis=1)
        
        if (opt):
            data_df = count_optimized(data_cont, row_sums, col_sums)
        else:
            data_df = count(data_cont, row_sums, col_sums)
        total_report_number = data_df['events'].sum()

    else:
        # Compute the flattened table directly from the data
        # This is how OpenEBGM makes counts
    
        actual = data_frame.groupby([product_label, ae_label])[id_label].nunique().reset_index(name='events') # number of times this product/ae pair occurs
        product_marg = data_frame.groupby([product_label])[id_label].nunique().reset_index(name='product_aes') # number of times this product appears
        ae_marg = data_frame.groupby([ae_label])[id_label].nunique().reset_index(name='count_across_brands') # number of times this ae appears 
        
        data_df = actual.merge(product_marg, on=product_label, how='inner')
        data_df = data_df.merge(ae_marg, on=ae_label, how='inner')
        data_df = data_df[[ 'events', 'product_aes', 'count_across_brands', ae_label, product_label]]
        total_report_number = data_frame[id_label].nunique()

    # Initialize the container object and assign the data
    DC = Container()
    DC.contingency = data_cont
    DC.data = data_df
    DC.N = total_report_number
    return DC



def compute_contingency(
    data_frame, product_label, count_label, ae_label, margin_threshold
):
    """Compute the contingency table for DA

    Args:
        data_frame (pd.DataFrame): A count data dataframe of the drug/device and events data
        product_label (str): Label of the column containing the product names
        count_label (str): Label of the column containing the event counts
        ae_label (str): Label of the column containing the adverse event counts
        margin_threshold (int): The minimum number of events required to keep a drug/device-event pair.

    Returns:
        pd.DataFrame: A contingency table with adverse events as columns and products as rows.
    """
    # Create a contingency table based on the brands and AEs
    data_cont = pd.pivot_table(
        data_frame,
        values=count_label,
        index=product_label,
        columns=ae_label,
        aggfunc="sum",
        fill_value=0,
    )

    # Calculate empty rows/columns based on margin_threshold and remove
    cut_rows = np.where(np.sum(data_cont, axis=1) < margin_threshold)
    drop_rows = data_cont.index[cut_rows]

    cut_cols = np.where(np.sum(data_cont, axis=0) < margin_threshold)
    drop_cols = data_cont.columns[cut_cols]

    data_cont = data_cont.drop(drop_rows)
    data_cont = data_cont.drop(drop_cols, axis=1)
    return data_cont


def convert_binary(data, product_label="name", ae_label="AE"):
    """Convert input data consisting of unique product-event pairs into a
       binary dataframe indicating which event and which product are
       associated with each other.

    Args:
        data (pd.DataFrame): A DataFrame consisting of unique product-event pairs for each row
        product_label (str, optional): If the product name is not in a column called `name`, override here. Defaults to "name".
        ae_label (str, optional): If the adverse event is not in a column called `AE`, override here.. Defaults to "AE".

    Returns:
        Container: A container with two binary dataframes. One is the X data of product names and the other is the
        y data with adverse events. Index locations are associated with the input DataFrame.

    """
    DC = Container()
    prod_df = pd.get_dummies(data[product_label], prefix="", prefix_sep="")
    DC.product_features = prod_df.groupby(by=prod_df.columns, axis=1).sum()

    event_df = pd.get_dummies(data[ae_label], prefix="", prefix_sep="")
    DC.event_outcomes = event_df.groupby(by=event_df.columns, axis=1).sum()
    DC.N = data.shape[0]
    return DC


def convert_multi_item(df, product_cols=["name"], ae_col="AE", min_threshold=3):
    """***WARNING*** Currently experimental and not guaranteed to perform as expected.
    Convert data with multiple product columns into a multi-item flattened dataframe for the DA methods.

    Args:
        df (pd.DataFrame): A dataframe where each row is a unique adverse event and has multiple columns
        indicating the presence of multiple devices/drugs/interventions.
        product_cols (list, optional): A list of column names associated with the co-occuring products. Defaults to ["name"].
        ae_col (str, optional): The column name that contains the adverse events. Defaults to "AE".
        min_threshold (int, optional): The minimum number of events required to keep a drug/device-event pair.

    Returns:
        Container: A container object that holds the necessary components for DA.
    """
    ae_counts = Counter()
    product_counts = Counter()
    for col in product_cols:
        product_counts.update(Counter(df[col]))
        ae_counts.update(Counter(df.loc[df[col] != ""][ae_col]))

    # Initialize an empty list to store the result
    result = []

    # Iterate over each row in the dataframe
    for _, row in df.iterrows():
        # Extract product names from the current row
        names = {row[x] for x in product_cols if row[x]}
        # Get all unique combinations of names (without repetition)
        combos = list(
            chain.from_iterable(
                combinations(names, r) for r in range(1, len(names) + 1)
            )
        )
        # Append combinations to the result list with the other column info
        for combo in combos:
            new_data = {idx: row[idx] for idx in row.index if idx not in product_cols}
            new_data["product_name"] = f"{'|'.join([c for c in combo if c])}"
            new_data["product_aes"] = sum([product_counts[p] for p in combo])
            new_data["count_across_brands"] = ae_counts[row[ae_col]]
            result.append(new_data)

    # Convert the result list to a new dataframe
    new_df = pd.DataFrame(result)
    event_series = new_df.groupby(by=["AE", "product_name"]).sum()["count"]
    new_df["events"] = new_df.apply(
        lambda x: event_series[x["AE"]][x["product_name"]], axis=1
    )
    new_df.rename(columns={ae_col: "ae_name"}, inplace=True)

    DC = Container()
    DC.contingency = compute_contingency(
        new_df, "product_name", "count", "ae_name", min_threshold
    )
    DC.data = new_df[
        ["ae_name", "product_name", "count_across_brands", "product_aes", "events"]
    ].drop_duplicates()
    DC.N = new_df["events"].sum()

    return DC


def count(data, rows, cols):
    """
    Convert the input contingency table to a flattened table

    Arguments:
        data (Pandas DataFrame): A contingency table of brands and events

    Returns:
        df: A Pandas DataFrame with the count information

    """
    d = {
        "events": [],
        "product_aes": [],
        "count_across_brands": [],
        "ae_name": [],
        "product_name": [],
    }
    for col, row in product(data.columns, data.index):
        n11 = data[col][row]
        if n11 > 0:
            d["count_across_brands"].append(cols[col])
            d["product_aes"].append(rows[row])
            d["events"].append(n11)
            d["product_name"].append(row)
            d["ae_name"].append(col)

    df = pd.DataFrame(d)
    return df

def count_optimized(data, rows, cols):
    """
    Convert the input contingency table to a flattened table
    This has been optimised using numpy functions to work efficiently on larger datasets

    Arguments:
        data (Pandas DataFrame): A contingency table of brands and events
        rows (Series) : A set of data points showing how often each product from the table appears 
        cols (Series) : A set of data points showing how often each event from the table appears

    Returns:
        df: A Pandas DataFrame with the count information
    """
    # Flatten the DataFrame and reset the index
    df = data.stack().reset_index()
    
    # Rename the columns
    df.columns = ['product_name', 'ae_name', 'events']
    
    # Filter out rows where events are zero
    df = df[df['events'] > 0]
    
    # Map the rows and columns to their respective names
    df['product_aes'] = df['product_name'].map(rows)
    df['count_across_brands'] = df['ae_name'].map(cols)

    # Reset the index
    df = df.reset_index()

    return df

def convert_multi_item_pipeline(
    input_data, 
    id_label='id',
    prod_label='name',
    ae_label='ae', 
    count_label='count',
    limit_tuple_size=True, 
    tuple_size_limit=2,
    count_unique_ids=True,
    opt_count=True):
    """
    Processes and transforms input data to generate counts of drug combinations based on specified parameters.

    This function performs the following steps:
    1. Removes duplicate rows based on the specified columns (`id_label`, `prod_label`, `ae_label`).
    2. Processes the deduplicated data to generate and filter drug combinations, converting them into a format suitable for counting.
    3. Renames the column containing combinations to match the `prod_label`.
    4. Uses the `convert` function to count the occurrences of each combination, considering options for counting unique IDs and optimization.

    Parameters:
    input_data (pd.DataFrame): The input DataFrame containing data with columns for identifiers, product names, and adverse events.
    id_label (str): The column name to use as the identifier for grouping. Default is 'id'.
    prod_label (str): The column name for product names. Default is 'name'.
    ae_label (str): The column name for adverse event labels. Default is 'ae'.
    count_label (str): The column name that contains the counts 
    limit_tuple_size (bool): Whether to limit the size of the combinations to a specified maximum. Default is True.
    tuple_size_limit (int): The maximum size of the combinations to keep if `limit_tuple_size` is True. Default is 2.
    count_unique_ids (bool): Whether to count occurrences of each combination based on unique identifiers. Default is True.
    opt_count (bool): An optimization flag for the `convert` function, affecting how counts are computed. Default is True.

    Returns:
    pd.DataFrame: A DataFrame with the counts of each drug combination. The DataFrame includes columns for:
                  - The product label (`prod_label`), representing the drug combinations.
                  - A count of occurrences.
                  - Adverse event labels and identifiers as per the `ae_label` and `id_label`.

    Notes:
    - The function relies on the `multi_item_processing` function to generate and filter drug combinations.
    - The `convert` function is used to count the occurrences of each combination, with options to count unique IDs and optimize the count calculation.
    """
    # read in input data and make sure to delete fully duplicated rows 
    dedup_df = input_data.drop_duplicates(subset=[id_label, prod_label, ae_label], keep='first')
    # process data to deal with tuple list
    df_proc = multi_item_processing(dedup_df, id_label, prod_label, ae_label, count_label, limit_tuple_size, tuple_size_limit)

    df_proc.rename(columns={
        'combinations': prod_label
    }, inplace=True)

    # use existing convert item to count this dataset
    cont = convert(df_proc, product_label=prod_label, count_label=count_label, ae_label=ae_label, count_unique_ids=count_unique_ids, id_label=id_label, opt=opt_count)

    return cont 

def get_permutations(row, limit_tuple_sizes=True, tuple_size_limit=2):
    """
    Generate all possible permutations (combinations) of drug names from a given row.

    This function processes a row from a DataFrame to extract drug names, generate all possible 
    combinations of these drugs, and optionally filter these combinations based on a maximum size limit.

    Parameters:
    row (pd.Series): A row from a DataFrame where drug names are stored in columns 
                     that start with 'Drug'. This row is expected to include drug columns 
                     with non-null values.
    limit_tuple_sizes (bool, optional): Whether to limit the size of the permutations to 
                                         a specified maximum. Default is True, meaning we are limiting sizes
    tuple_size_limit (int, optional): The maximum size of the permutations to keep if 
                                       `limit_tuple_sizes` is True. Default is 2

    Returns:
    list of tuples: A list containing tuples of drug names. Each tuple represents a combination 
                    of drug names, with the size of the tuples optionally limited by 
                    `tuple_size_limit`.

    Notes:
    - Drug columns in the row must have names starting with 'Drug' for extraction.
    - Permutations are generated in lexicographic order because the drug names are sorted alphabetically 
      before generating combinations.
    - If `limit_tuple_sizes` is True, only combinations with up to `tuple_size_limit` elements are included.
    """
    # Extract non-null drug columns for the current row, ignoring None values
    drugs = [row[col] for col in row.index if col.startswith('Drug') and row[col] is not None and row[col] != '']
    
    # Sort drug columns alphabetically for consistently
    drugs.sort()
    
    # List to store all combinations (individual drugs and their combinations)
    all_combinations = []
    
    # Generate all possible combinations of 1, 2, ... len(drugs)
    for r in range(1, len(drugs) + 1):
        if limit_tuple_sizes and tuple_size_limit is not None and r > tuple_size_limit:
            break
        all_combinations.extend(combinations(drugs, r))
    
    return all_combinations

def multi_item_processing(
    input_data, 
    id_label='id', 
    prod_label='name', 
    ae_label='AE',
    count_label='count', 
    limit_tuple_sizes=True, 
    tuple_size_limit=2):
    """
    Processes a DataFrame to generate and format all possible permutations of drug names grouped by 
    specified identifiers and conditions. 

    The algorithm here is:
    1) we start off with table where each column has an id (id_label), an adverse event (ae_label) and a product (prod_label). The nature of SRS means that you have multiple rows with the same 
    id/AE, but different drugs. We want a way to process this to be able to produce a table that takes into account 
    potential pairs of drugs (and higher order effects). 
    2) We create a series of columns, all of which start with the string Drug to show all of the different drugs that occur on a single AE/id combo. 
    3) We then calculate all the possible combinations of the variables in these columns, creating it as a new column. 
    4) Optionally, we limit the size of the combinations at this point to 2 (because higher order interactions mean drastically many more combinations in the table) 
    5) We then explode out this column, to produce the necessary drugs and drug pairs that correspond to each id and adverse event as a new column

    Parameters:
    input_data (pd.DataFrame): The DataFrame containing the data to process. It should include columns 
                               for identifiers, product names, and adverse events.
    id_label (str): The name of the column to use as the identifier for grouping. Default is 'id'.
    prod_label (str): The name of the column containing product names. Default is 'name'.
    ae_label (str): The name of the column containing adverse event labels. Default is 'ae'.
    count_label(str): The name you are going to name the 'count' column
    limit_tuple_sizes (bool): Whether to limit the size of tuples in the combinations. Default is True.
    tuple_size_limit (int): The maximum size of tuples to keep if `limit_tuple_sizes` is True. Default is 2.

    Returns:
    pd.DataFrame: A DataFrame with columns `id`, `ae`, `combinations`, and `count`. 
                  - `id`: Identifier used for grouping.
                  - `ae`: Adverse event label.
                  - `combinations`: A string of drug combinations separated by '|'.
                  - `count`: A count column with a constant value of 1.
    """

    # Combine drug names within each group
    df_combined = (
    input_data.groupby([id_label, ae_label])[prod_label]
    .apply(lambda x: pd.Series(x.to_numpy()))
    .unstack()
    .reset_index()
    )

    # Rename the columns
    df_combined.columns = [id_label, ae_label] + [f'Drug{i+1}' for i in range(df_combined.shape[1] - 2)]

    drug_columns = [col for col in df_combined.columns if col.startswith('Drug')]
    df_combined[drug_columns] = df_combined[drug_columns].where(pd.notnull(df_combined[drug_columns]), None)

    # Generate permutations
    df_combined['combinations'] = df_combined.apply(
        lambda row: get_permutations(row, limit_tuple_sizes, tuple_size_limit), 
        axis=1
    )

    # Filter out rows where the permutations list is empty
    df_combined = df_combined[df_combined['combinations'].apply(lambda x: len(x) > 0)]
    
    # Explode the filtered permutations into separate rows
    df_exploded = df_combined.explode('combinations').reset_index(drop=True)

    # Clean up the dataframe
    df_cleaned = df_exploded.drop(columns=df_exploded.filter(like='Drug').columns)
    df_cleaned[count_label] = 1

    # Convert tuples to single items separated by '|'
    df_cleaned['combinations'] = df_cleaned['combinations'].apply(lambda x: '    |    '.join(x))

    return df_cleaned


