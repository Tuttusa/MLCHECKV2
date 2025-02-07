#!/usr/bin/env python


def funcWriteXml(df):
    """
    Generate XML specification file for MLCheck matching the required format
    
    Args:
        df: pandas DataFrame containing the dataset
    """
    with open('dataInput.xml', 'w') as f:
        # Start directly with Inputs tag - removing XML declaration
        f.write('<Inputs>\n')
        
        # Process only feature columns (exclude 'Class' if present)
        feature_columns = [col for col in df.columns if col != 'Class']
        
        for col in feature_columns:
            f.write('<Input>\n')
            # Convert column names to match expected format
            col_name = col.replace('-', '_')  # Replace hyphens with underscores
            f.write(f'<Feature-name>{col_name}</Feature-name>\n')
            f.write('<Feature-type>int64</Feature-type>\n')
            f.write('<Value>\n')
            
            # Convert to integer values
            min_val = int(df[col].min())
            max_val = int(df[col].max())
            
            f.write(f'<minVal>{min_val}</minVal>\n')
            f.write(f'<maxVal>{max_val}</maxVal>\n')
            f.write('</Value>\n')
            f.write('</Input>\n')
        
        f.write('</Inputs>')


# df = pd.read_csv(str(sys.argv[1]))
# funcWriteXml(df)


# fe_dict = {}
# for i in range(0, df.shape[1]):
#     fe_dict[df.columns.values[i]] = str(df.dtypes[i])

# try:
#     with open('feNameType.csv', 'w') as csv_file:
#         writer = cv.writer(csv_file)
#         for key, value in fe_dict.items():
#             writer.writerow([key, value])
# except IOError:
#     print("I/O error")
