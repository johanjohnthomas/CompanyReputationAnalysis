import pandas as pd
import sys
"""
This script reads a CSV file containing posts and comments, 
and splits it into two separate CSV files: 
one containing only the comments and the other containing only the posts.

This is what we used to split the comments and the posts for both the datasets.
"""
def split_csv(input_file, comment_file, post_file):
    # Read the input CSV file
    df = pd.read_csv(input_file)

    # Split the dataframe based on the 'type' column
    comments_df = df[df['type'] == 'comment']
    posts_df = df[df['type'] == 'post']

    # Write the dataframes to separate CSV files
    comments_df.to_csv(comment_file, index=False)
    posts_df.to_csv(post_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python splitDataset.py <input_file> <comment_file> <post_file>")
        sys.exit(1)
    # Retreive arguments from command line
    input_file = sys.argv[1]
    comment_file = sys.argv[2]
    post_file = sys.argv[3]

    split_csv(input_file, comment_file, post_file)