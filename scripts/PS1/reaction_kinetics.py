import matplotlib.pyplot as plt
import pandas as pd

# Declare global data dictionary to hold time and concentration data

data = {"t (s)": [], "A (uM)": [], "B (uM)": []}


def generateReactionData(file_in: str, file_out: str) -> list:
    """
    Calculates and graphs the formation of the product AB from reaction data, writes to
    a new output file the original data plus an AB column.
    :param file_in: Reaction kinetics data file
    :param file_out: Name of output file
    :returns list of AB concentration over time
    """
    with open(file_in, "r") as file:  # read in the file
        lines = file.readlines()
        A0 = float(lines[1].strip().split()[1])  # remove the whitespace and special characters, store A0

        # loop over the lines after the header, appending the appropriate columns to the data object
        for line in lines[1:]:
            data["t (s)"].append(float(line.strip().split()[0]))
            data["A (uM)"].append(float(line.strip().split()[1]))
            data["B (uM)"].append(float(line.strip().split()[2]))

    # calculate AB = A0 - A(t), pass to dataframe for plotting
    data["AB (uM)"] = [A0 - a for a in data["A (uM)"]]
    df = pd.DataFrame(data)

    # plotting
    df.plot(x="t (s)")
    plt.xlim(0, 10)
    plt.ylabel("Concentration (uM)")
    plt.title("Concentration as A Function of Time: A+B⇌AB")

    # write to new output file
    with open(file_out, "w") as file:
        file.write(df.to_string(index=False))

    return data["AB (uM)"]


# Example usage (not called when imported)
if __name__ == "__main__":
    ab = generateReactionData("../../data/A_B_vs_time.dat", "../../data/A_B_product.dat")