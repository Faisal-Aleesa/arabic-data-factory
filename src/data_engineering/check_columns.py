from datasets import load_dataset

ds = load_dataset("mohres/The_Arabic_E-Book_Corpus", split="train")
print("Columns:", ds.column_names)
print("\nFirst row sample:")
print(ds[0])