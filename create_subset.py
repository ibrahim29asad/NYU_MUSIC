import os

input_file = "train_scaling.txt" 
output_file = "train_100M.txt"    

M_size = 100 * 1024 * 1024 

print(f"Reading from {input_file}...")

try:
    with open(input_file, "r", encoding="utf-8") as f_in:
        chunk = f_in.read(M_size)
    with open(output_file, "w", encoding="utf-8") as f_out:
        f_out.write(chunk)
    print("Dones")
except FileNotFoundError:
    print(f"No")