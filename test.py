import pandas as pd
apple = ['[8.5, 9.5, 9, 8, 9.5]', '[2, 9, 6, 4, 6]', '[4, 8, 3, 5, 6]'] 

def extract_inner_lists(data_list):
  """Extracts the inner lists from a list containing strings of lists.

  Args:
      data_list: A list containing strings that represent lists.

  Returns:
      A list containing the extracted inner lists.
  """
  inner_lists = []
  for item in data_list:
    # Try converting the string to a list using eval
    try:
      inner_list = eval(item)
    except (SyntaxError, NameError):
      # Handle potential errors during evaluation
      print(f"Error: Could not evaluate string '{item}' as a list")
      continue
    # Check if the evaluated item is a list
    if isinstance(inner_list, list):
      inner_lists.append(inner_list)
  return inner_lists

inner_lists = extract_inner_lists(apple)
print(inner_lists)
metric_names = ["Accuracy (0-10)", "Clarity (0-10)", "Conciseness (0-10)","Depth (0-10)","Relevance (0-10)"]  # List of metric names
df = pd.DataFrame(inner_lists, columns=metric_names)
print(df)
