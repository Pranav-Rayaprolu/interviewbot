import pandas as pd
df = pd.read_csv('all_companies2.csv')
free_companies = ["Wipro","Infosys", "TCS" ,"HCL Technologies","KPIT Tech"]
free_positions = ["Machine Learning", "NLP", "Python" ]
available_positions = df[['company_name'.isin(free_companies), 'Job_Position'.isin(free_positions)]].apply(lambda x: (x['company_name'].lower(), x['Job_Position'].lower()), axis=1).tolist()
print(available_positions)

# import pandas as pd
# df = pd.read_csv('all_companies2.csv')
# def identify_available_positions(df, free_companies, free_positions):
#   """
#   Identifies available positions from a DataFrame based on company and position criteria.

#   Args:
#       df (pd.DataFrame): The DataFrame containing company and job position data.
#       free_companies (list): A list of companies considered "free" (open to new opportunities).
#       free_positions (list): A list of job positions that are considered "free" (openings).

#   Returns:
#       list: A list of tuples, where each tuple contains the lowercase company name
#             and lowercase job position for companies that have at least one
#             matching free position.
#   """

#   # Ensure company and job position columns exist in the DataFrame
#   if 'company_name' not in df.columns or 'Job_Position' not in df.columns:
#     raise ValueError("Required columns 'company_name' and 'Job_Position' not found in the DataFrame.")

#   # Create boolean masks for matching companies and positions
#   company_mask = df['company_name'].isin(free_companies)
#   position_mask = df['Job_Position'].str.lower().isin(free_positions)  # Efficient lowercase conversion

#   # Combine masks using logical AND to ensure companies have at least one matching position
#   combined_mask = company_mask & position_mask

#   # Extract relevant data for matching rows
#   available_positions = df[combined_mask][['company_name', 'Job_Position']].to_list()  # Use to_list()

#   return available_positions

# # Assuming you have your DataFrame 'df' loaded from 'all_companies2.csv'
# free_companies = ["Wipro", "Infosys", "TCS", "HCL Technologies", "KPIT Tech"]
# free_positions = ["Machine Learning", "NLP", "Python"]

# available_positions = identify_available_positions(df, free_companies, free_positions)
# print(available_positions)

