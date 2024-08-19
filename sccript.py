import os

def main():
    with open("data.txt", "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk("."):
            for filename in files:
                if filename.endswith(".jl"):
                    filepath = os.path.join(root, filename)
                    print(f"Processing: {filepath}")  # Add print statement here
                    outfile.write(f"[{filepath}]\n")
                    with open(filepath, "r", encoding="utf-8") as infile:
                        outfile.write(infile.read())
                    outfile.write("\n")

if __name__ == "__main__":
    main()